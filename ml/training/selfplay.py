"""
Self-Play Game Generation for AlphaZero-Style Training

This module implements the self-play engine that generates training data by
playing games using MCTS with the current neural network. The generated games
provide (state, policy, value) tuples for supervised learning.

Core Concept:
    - Generate games using current model + MCTS
    - Store (state, MCTS_policy, final_outcome) tuples for training
    - Use exploration noise to ensure diverse gameplay
    - Run multiple games in parallel for efficiency

Game Generation Flow:
    1. Initialize game with random setup
    2. For each decision point (bid or card play):
        - Run MCTS with current model
        - Get action probabilities (visit counts)
        - Sample action with temperature-based exploration
        - Store (state, policy, None) for training
    3. Play until game ends
    4. Back-propagate final outcome to all stored positions
    5. Return training examples

Key Design Decisions:
    - Use imperfect info MCTS (realistic hidden information)
    - Temperature schedule: high early (exploration), low late (exploitation)
    - Store full game history for later analysis
    - Support variable player counts (3-8 players)

Training Example Format:
    {
        'state': encoded_state,          # 256-dim numpy array
        'policy': action_probabilities,  # MCTS visit counts (52-dim, matches network)
        'value': final_score,            # Outcome from this player's perspective
        'player_position': int,          # Which player made this decision
        'game_id': str,                  # For tracking game history
        'move_number': int,              # Position in game
    }
"""

import torch
import numpy as np
import uuid
import json
import time as _time
import multiprocessing as mp
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.game.blob import BlobGame, Card
from ml.mcts.search import ImperfectInfoMCTS


class SelfPlayWorker:
    """
    Worker that generates self-play games for training.

    Runs MCTS-guided game generation and collects training examples.
    Each decision point in the game produces a training example with:
    - Current state encoding
    - MCTS policy (action probabilities)
    - Final game outcome (back-propagated after game ends)
    """

    def __init__(
        self,
        network: BlobNet,
        encoder: StateEncoder,
        masker: ActionMasker,
        num_determinizations: int = 3,
        simulations_per_determinization: int = 30,
        temperature_schedule: Optional[Callable[[int], float]] = None,
        use_imperfect_info: bool = True,
        batch_evaluator: Optional["BatchedEvaluator"] = None,
        gpu_server_client: Optional["GPUServerClient"] = None,
        batch_size: Optional[int] = None,
        use_parallel_expansion: bool = False,
        parallel_batch_size: int = 10,
        must_have_suit_bias: float = 1.0,
    ):
        """
        Initialize self-play worker.

        Args:
            network: Neural network for MCTS
            encoder: State encoder
            masker: Action masker
            num_determinizations: Determinizations per MCTS search (default: 3)
            simulations_per_determinization: MCTS simulations per world (default: 30)
            temperature_schedule: Function mapping move_number -> temperature
                                 If None, uses default schedule
            use_imperfect_info: Use imperfect info MCTS (vs perfect info)
            batch_evaluator: Optional BatchedEvaluator for multi-game batching
                            If provided, uses centralized batching for better GPU utilization
            gpu_server_client: Optional GPUServerClient for multiprocessing GPU server
                              If provided, sends requests to centralized GPU server process
            batch_size: Optional batch size for Phase 1 intra-game batching
                       If None, uses sequential search() for baseline
                       If set, uses search_batched() with virtual loss mechanism
            use_parallel_expansion: Use GPU-batched parallel expansion (default: False)
                                   If True, uses search_parallel() for cross-worker batching
                                   Requires batch_evaluator or gpu_server_client
            parallel_batch_size: Batch size for parallel expansion (default: 10)
                                Number of leaves to expand per iteration
                                Recommended: 10 for 32 workers (32×10=320 batch size)
            must_have_suit_bias: Probability multiplier for must-have suits during determinization (default: 1.0)
                                1.0 = no bias (maximum entropy), higher values = stronger preference
        """
        self.network = network
        self.encoder = encoder
        self.masker = masker
        self.num_determinizations = num_determinizations
        self.simulations_per_determinization = simulations_per_determinization
        self.use_imperfect_info = use_imperfect_info
        self.batch_evaluator = batch_evaluator
        self.gpu_server_client = gpu_server_client
        self.batch_size = batch_size
        self.use_parallel_expansion = use_parallel_expansion
        self.parallel_batch_size = parallel_batch_size
        self.must_have_suit_bias = must_have_suit_bias

        # Temperature schedule for exploration
        if temperature_schedule is None:
            self.temperature_schedule = self.get_default_temperature_schedule()
        else:
            self.temperature_schedule = temperature_schedule

        # Create MCTS instance
        if use_imperfect_info:
            self.mcts = ImperfectInfoMCTS(
                network=network,
                encoder=encoder,
                masker=masker,
                num_determinizations=num_determinizations,
                simulations_per_determinization=simulations_per_determinization,
                batch_evaluator=batch_evaluator,
                gpu_server_client=gpu_server_client,
                must_have_suit_bias=must_have_suit_bias,
            )
        else:
            # For testing or comparison, can use perfect info MCTS
            from ml.mcts.search import MCTS

            total_sims = num_determinizations * simulations_per_determinization
            self.mcts = MCTS(
                network=network,
                encoder=encoder,
                masker=masker,
                num_simulations=total_sims,
                batch_evaluator=batch_evaluator,
                gpu_server_client=gpu_server_client,
            )

    def generate_game(
        self,
        num_players: int = 4,
        cards_to_deal: int = 5,
        game_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a single self-play game.

        Plays a complete game using MCTS for all decisions, collecting training
        examples at each decision point. Final outcomes are back-propagated to
        all examples after the game completes.

        Args:
            num_players: Number of players in the game (3-8)
            cards_to_deal: Cards to deal per player (1-13)
            game_id: Optional game identifier (generates UUID if None)

        Returns:
            List of training examples (one per decision point)
            Each example is a dict with keys: state, policy, value,
            player_position, game_id, move_number
        """
        if game_id is None:
            game_id = str(uuid.uuid4())

        # Initialize game
        game = BlobGame(num_players=num_players)

        # Storage for training examples
        examples = []
        move_number = 0

        # Play the round using MCTS for all decisions
        def get_bid(player, hand, is_dealer, total_bids, cards_dealt):
            """Callback to get bid using MCTS."""
            nonlocal move_number

            # Run MCTS to get action probabilities
            # GPU-Batched: Use search_parallel() if use_parallel_expansion is True
            # Phase 1: Use search_batched() if batch_size is set
            # Baseline: Use search() if batch_size is None
            if self.use_parallel_expansion:
                action_probs = self.mcts.search_parallel(
                    game, player, parallel_batch_size=self.parallel_batch_size
                )
            elif self.batch_size is not None:
                action_probs = self.mcts.search_batched(
                    game, player, batch_size=self.batch_size
                )
            else:
                action_probs = self.mcts.search(game, player)

            # Get temperature for this move
            temperature = self.temperature_schedule(move_number)

            # Select action with temperature
            bid = self._select_action(action_probs, temperature)

            # Store training example (value will be filled in later)
            state_tensor = self.encoder.encode(game, player)
            policy_vector = self._action_probs_to_vector(action_probs, is_bidding=True)

            examples.append(
                {
                    "state": state_tensor.cpu().numpy(),
                    "policy": policy_vector,
                    "value": None,  # Will be back-propagated
                    "player_position": player.position,
                    "game_id": game_id,
                    "move_number": move_number,
                }
            )

            move_number += 1
            return bid

        def get_card(player, legal_cards, trick):
            """Callback to get card to play using MCTS."""
            nonlocal move_number

            # Run MCTS to get action probabilities
            # GPU-Batched: Use search_parallel() if use_parallel_expansion is True
            # Phase 1: Use search_batched() if batch_size is set
            # Baseline: Use search() if batch_size is None
            if self.use_parallel_expansion:
                action_probs = self.mcts.search_parallel(
                    game, player, parallel_batch_size=self.parallel_batch_size
                )
            elif self.batch_size is not None:
                action_probs = self.mcts.search_batched(
                    game, player, batch_size=self.batch_size
                )
            else:
                action_probs = self.mcts.search(game, player)

            # Get temperature for this move
            temperature = self.temperature_schedule(move_number)

            # Select action with temperature
            card_idx = self._select_action(action_probs, temperature)

            # Find the card in legal_cards that matches the selected index
            card = None
            for legal_card in legal_cards:
                if self.encoder._card_to_index(legal_card) == card_idx:
                    card = legal_card
                    break

            # Fallback to first legal card if MCTS selected an illegal card
            if card is None:
                card = legal_cards[0]

            # Store training example
            state_tensor = self.encoder.encode(game, player)
            policy_vector = self._action_probs_to_vector(
                action_probs, is_bidding=False
            )

            examples.append(
                {
                    "state": state_tensor.cpu().numpy(),
                    "policy": policy_vector,
                    "value": None,  # Will be back-propagated
                    "player_position": player.position,
                    "game_id": game_id,
                    "move_number": move_number,
                }
            )

            move_number += 1
            return card

        # Play the round
        result = game.play_round(cards_to_deal, get_bid, get_card)

        # Back-propagate final outcomes to all examples
        # Extract round scores from result
        final_scores = {}
        for player_result in result["player_scores"]:
            # Find the player by name to get their position
            for player in game.players:
                if player.name == player_result["name"]:
                    final_scores[player.position] = player_result["round_score"]
                    break

        self._backpropagate_outcome(examples, final_scores)

        return examples

    def _select_action(
        self,
        action_probs: Dict[int, float],
        temperature: float,
    ) -> int:
        """
        Select action using temperature-based sampling.

        Temperature controls exploration vs exploitation:
        - temperature = 0: Greedy (always pick best action)
        - temperature = 1: Proportional to probabilities
        - temperature > 1: More exploration

        Args:
            action_probs: Action probabilities from MCTS (dict: action_idx -> prob)
            temperature: Exploration temperature (0=greedy, 1=stochastic)

        Returns:
            Selected action index
        """
        if len(action_probs) == 0:
            raise ValueError("No legal actions available")

        # Extract actions and probabilities
        actions = list(action_probs.keys())
        probs = np.array([action_probs[a] for a in actions])

        # Apply temperature
        if temperature == 0:
            # Greedy: select action with highest probability
            best_idx = np.argmax(probs)
            return actions[best_idx]
        else:
            # Apply temperature scaling
            # Higher temperature = more uniform distribution
            # Lower temperature = more peaked distribution
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()  # Renormalize

            # Sample action
            selected_idx = np.random.choice(len(actions), p=probs)
            return actions[selected_idx]

    def _action_probs_to_vector(
        self, action_probs: Dict[int, float], is_bidding: bool
    ) -> np.ndarray:
        """
        Convert action probabilities dict to fixed-size vector.

        Creates a 52-dimensional vector to match network output dimension.
        For bidding: indices 0-13 are bids
        For playing: indices 0-51 are card indices

        Args:
            action_probs: Dict mapping action_idx -> probability
            is_bidding: True if this is a bidding action

        Returns:
            52-dim numpy array with probabilities (zeros for illegal actions)
        """
        # Use 52 dimensions to match network action_dim = max(14, 52) = 52
        policy_vector = np.zeros(52, dtype=np.float32)

        for action_idx, prob in action_probs.items():
            if action_idx < 52:  # Safety check
                policy_vector[action_idx] = prob

        return policy_vector

    def _backpropagate_outcome(
        self,
        examples: List[Dict[str, Any]],
        final_scores: Dict[int, int],
    ):
        """
        Back-propagate final game outcome to all training examples.

        Updates the 'value' field of each example with the final score
        achieved by the player who made that decision.

        Value normalization:
        - Scores in Blob range from 0 (failed bid) to 23 (bid 13 and made it)
        - Normalize to [-1, 1] range for neural network training
        - 0 points -> -1.0, 23 points -> 1.0

        Args:
            examples: List of training examples from the game
            final_scores: Final scores for each player {position: score}
        """
        # Find max possible score for normalization
        # Max score is 10 + 13 = 23 (bid 13 and make it)
        max_score = 23.0

        for example in examples:
            player_position = example["player_position"]
            score = final_scores.get(player_position, 0)

            # Normalize score to [-1, 1] range
            # 0 -> -1.0, 23 -> 1.0
            normalized_value = (score / max_score) * 2.0 - 1.0

            example["value"] = float(normalized_value)

    def get_default_temperature_schedule(self) -> Callable[[int], float]:
        """
        Get default temperature schedule.

        Temperature controls exploration during action selection:
        - Early game (moves 0-10): temperature = 1.0 (high exploration)
        - Mid game (moves 11-20): temperature = 0.5 (moderate)
        - Late game (moves 21+): temperature = 0.1 (near-greedy)

        This encourages diverse play early (to explore different strategies)
        and more deterministic play later (to demonstrate learned skill).

        Returns:
            Function mapping move_number -> temperature
        """

        def schedule(move_number: int) -> float:
            if move_number < 10:
                return 1.0  # High exploration
            elif move_number < 20:
                return 0.5  # Moderate exploration
            else:
                return 0.1  # Near-greedy

        return schedule


class SelfPlayEngine:
    """
    Manages parallel self-play game generation.

    Orchestrates multiple workers to generate games efficiently using either:
    - ThreadPoolExecutor (Phase 3): Shared BatchedEvaluator for GPU batching
    - multiprocessing.Pool (Phase 2): Isolated workers with per-worker batching

    Architecture (Phase 3 - ThreadPoolExecutor):
        - Main thread creates shared BatchedEvaluator
        - Worker threads share single network and evaluator instance
        - All MCTS instances send requests to same BatchedEvaluator
        - Large batch sizes (128-512+) maximize GPU utilization

    Architecture (Phase 2 - multiprocessing):
        - Main process distributes game generation tasks to worker pool
        - Each worker creates its own SelfPlayWorker with network copy
        - Workers run independently with different random seeds
        - Each worker has its own BatchedEvaluator (small batches)

    Performance:
        - Phase 3 (threads + shared evaluator): 1000-2000 games/min, 70-90% GPU
        - Phase 2 (processes): 96 games/min, 5-10% GPU (overhead dominates)
        - Phase 1 (sequential): 220 games/min, no batching
    """

    def __init__(
        self,
        network: BlobNet,
        encoder: StateEncoder,
        masker: ActionMasker,
        num_workers: int = 16,
        num_determinizations: int = 3,
        simulations_per_determinization: int = 30,
        temperature_schedule: Optional[Callable[[int], float]] = None,
        device: str = "cpu",
        use_batched_evaluator: bool = True,
        batch_size: int = 512,
        batch_timeout_ms: float = 10.0,
        use_thread_pool: Optional[bool] = None,
        use_gpu_server: bool = False,
        gpu_server_max_batch: int = 512,
        gpu_server_timeout_ms: float = 10.0,
        mcts_batch_size: Optional[int] = None,
        use_parallel_expansion: bool = True,
        parallel_batch_size: int = 10,
        enable_worker_profiling: bool = False,
        enable_worker_metrics: bool = False,
        run_id: Optional[str] = None,
        must_have_suit_bias: float = 1.0,
    ):
        """
        Initialize self-play engine.

        Args:
            network: Neural network for MCTS
            encoder: State encoder
            masker: Action masker
            num_workers: Number of parallel workers (default: 16)
            num_determinizations: Determinizations per MCTS search
            simulations_per_determinization: MCTS simulations per world
            temperature_schedule: Temperature schedule function
            device: Device for neural network ('cpu' or 'cuda')
            use_batched_evaluator: Use BatchedEvaluator for multi-game batching (default: True)
            batch_size: Maximum batch size for BatchedEvaluator (default: 512)
            batch_timeout_ms: Timeout in ms for batch collection (default: 10.0)
            use_thread_pool: Use ThreadPoolExecutor (Phase 3) instead of multiprocessing (Phase 2)
                            If None, auto-selects: True for GPU, False for CPU
            use_gpu_server: Use GPU inference server (Phase 3.5) for multiprocessing batching
                           Overrides use_batched_evaluator and use_thread_pool
            gpu_server_max_batch: Maximum batch size for GPU server (default: 512)
            gpu_server_timeout_ms: Batch collection timeout for GPU server (default: 10ms)
            mcts_batch_size: Phase 1 intra-game batching size (default: None for sequential)
                            If set, uses search_batched() with virtual loss
                            If None, uses search() for baseline
            use_parallel_expansion: Use parallel MCTS expansion for better batching (default: True)
            parallel_batch_size: Number of leaves to expand per iteration (default: 10)
            enable_worker_profiling: Enable cProfile profiling for worker 0 (default: False)
                                    When enabled, saves profiling data to profile_worker0.prof
                                    Useful for performance analysis but adds overhead (~5-10%)
            must_have_suit_bias: Probability multiplier for must-have suits during determinization (default: 1.0)
                                1.0 = no bias (maximum entropy), higher values = stronger preference
        """
        self.network = network
        self.encoder = encoder
        self.masker = masker
        self.num_workers = num_workers
        self.num_determinizations = num_determinizations
        self.simulations_per_determinization = simulations_per_determinization
        self.temperature_schedule = temperature_schedule
        self.device = device
        self.use_gpu_server = use_gpu_server
        self.gpu_server_max_batch = gpu_server_max_batch
        self.gpu_server_timeout_ms = gpu_server_timeout_ms
        self.mcts_batch_size = mcts_batch_size
        self.use_parallel_expansion = use_parallel_expansion
        self.parallel_batch_size = parallel_batch_size
        self.enable_worker_profiling = enable_worker_profiling
        self.enable_worker_metrics = enable_worker_metrics
        self.run_id = run_id or str(uuid.uuid4())
        self.must_have_suit_bias = must_have_suit_bias

        # GPU server takes precedence over other batching methods
        if use_gpu_server:
            self.use_batched_evaluator = False
            self.use_thread_pool = False  # Force multiprocessing
        else:
            self.use_batched_evaluator = use_batched_evaluator
            self.batch_size = batch_size
            self.batch_timeout_ms = batch_timeout_ms

            # Auto-select threading mode based on device
            if use_thread_pool is None:
                # Use threads for GPU (enables shared BatchedEvaluator)
                # Use processes for CPU (avoids GIL contention)
                self.use_thread_pool = (device == "cuda")
            else:
                self.use_thread_pool = use_thread_pool

        # Get network state dict for passing to workers (multiprocessing only)
        # Workers will create their own network instances from this
        self.network_state = network.state_dict() if not self.use_thread_pool else None

        # Parallel execution pools
        self.pool = None  # multiprocessing.Pool (Phase 2)
        self.thread_pool = None  # ThreadPoolExecutor (Phase 3)
        self.executor = None  # For async generation

        # GPU inference server (Phase 3.5)
        self.gpu_server = None
        self.gpu_clients = {}  # client_id -> GPUServerClient
        if use_gpu_server:
            from ml.mcts.gpu_server import GPUInferenceServer

            self.gpu_server = GPUInferenceServer(
                network=network,
                device=device,
                max_batch_size=gpu_server_max_batch,
                timeout_ms=gpu_server_timeout_ms,
            )
            self.gpu_server.start()
            print(f"[SelfPlayEngine] GPU server started (max_batch={gpu_server_max_batch}, timeout={gpu_server_timeout_ms}ms)")

        # Batched evaluator for multi-game batching
        self.batch_evaluator = None
        if self.use_batched_evaluator:
            from ml.mcts.batch_evaluator import BatchedEvaluator

            # For threads (Phase 3): Create shared evaluator
            # For processes (Phase 2): Create evaluator in main process (workers create their own)
            if self.use_thread_pool:
                # Phase 3: Single shared evaluator for all threads
                # This enables true cross-worker batching with large batch sizes
                self.batch_evaluator = BatchedEvaluator(
                    network=network,
                    max_batch_size=batch_size * 2,  # Larger for cross-worker batching
                    timeout_ms=batch_timeout_ms,
                    device=device,
                    enable_profiling_metrics=self.enable_worker_metrics,
                )
                self.batch_evaluator.start()
            else:
                # Phase 2: Multiprocess mode - workers create their own evaluators
                # Parent process doesn't need an evaluator (avoids idle thread overhead)
                # Each worker creates its own BatchedEvaluator in _worker_generate_games_static
                self.batch_evaluator = None

    def generate_games(
        self,
        num_games: int,
        num_players: int = 4,
        cards_to_deal: int = 5,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple games in parallel.

        Uses either ThreadPoolExecutor (Phase 3) or multiprocessing.Pool (Phase 2)
        depending on configuration. Thread pool enables shared BatchedEvaluator
        for maximum GPU utilization.

        Args:
            num_games: Total number of games to generate
            num_players: Players per game
            cards_to_deal: Cards to deal per player
            progress_callback: Optional callback(games_completed) for progress updates

        Returns:
            Flat list of all training examples from all games
        """
        if num_games <= 0:
            return []

        if self.use_thread_pool:
            # Phase 3: ThreadPoolExecutor with shared BatchedEvaluator
            return self._generate_games_threaded(
                num_games, num_players, cards_to_deal, progress_callback
            )
        else:
            # Phase 2: multiprocessing.Pool with per-worker BatchedEvaluator
            return self._generate_games_multiprocess(
                num_games, num_players, cards_to_deal, progress_callback
            )

    def _generate_games_multiprocess(
        self,
        num_games: int,
        num_players: int,
        cards_to_deal: int,
        progress_callback: Optional[Callable[[int], None]],
    ) -> List[Dict[str, Any]]:
        """Generate games using multiprocessing.Pool (Phase 2 or Phase 3.5)."""
        # Create process pool if not already created
        if self.pool is None:
            self.pool = mp.Pool(processes=self.num_workers)

        # Distribute games evenly across workers
        games_per_worker = num_games // self.num_workers
        remaining_games = num_games % self.num_workers

        # Different task creation based on GPU server usage
        if self.use_gpu_server:
            # Phase 3.5: Use GPU server with multiprocessing
            # Create clients for each worker
            tasks = []
            for worker_id in range(self.num_workers):
                worker_games = games_per_worker + (1 if worker_id < remaining_games else 0)
                if worker_games > 0:
                    # Create client for this worker
                    client_id = f"worker_{worker_id}"
                    client = self.gpu_server.create_client(client_id)
                    self.gpu_clients[client_id] = client

                    tasks.append(
                        (
                            worker_id,
                            worker_games,
                            num_players,
                            cards_to_deal,
                            client.request_queue,
                            client.response_queue,
                            client_id,
                            self.num_determinizations,
                            self.simulations_per_determinization,
                            self.temperature_schedule,
                            self.use_parallel_expansion,
                            self.parallel_batch_size,
                            self.enable_worker_profiling,
                            self.enable_worker_metrics,
                            self.run_id,
                            self.must_have_suit_bias,
                        )
                    )

            # Execute tasks with GPU server worker function
            results = self.pool.starmap(_worker_generate_games_with_gpu_server, tasks)

        else:
            # Phase 2: Standard multiprocessing with per-worker networks
            tasks = []
            for worker_id in range(self.num_workers):
                worker_games = games_per_worker + (1 if worker_id < remaining_games else 0)
                if worker_games > 0:
                    tasks.append(
                        (
                            worker_id,
                            worker_games,
                            num_players,
                            cards_to_deal,
                            self.network_state,
                            self.num_determinizations,
                            self.simulations_per_determinization,
                            self.temperature_schedule,
                            self.device,
                            self.use_batched_evaluator,
                            self.batch_size,
                            self.batch_timeout_ms,
                            self.mcts_batch_size,
                            self.use_parallel_expansion,
                            self.parallel_batch_size,
                            self.enable_worker_profiling,
                            self.enable_worker_metrics,
                            self.run_id,
                            self.must_have_suit_bias,
                        )
                    )

            # Execute tasks in parallel
            results = self.pool.starmap(_worker_generate_games_static, tasks)

        # Flatten results from all workers
        all_examples = []
        for worker_examples in results:
            all_examples.extend(worker_examples)

        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(num_games)

        return all_examples

    def _generate_games_threaded(
        self,
        num_games: int,
        num_players: int,
        cards_to_deal: int,
        progress_callback: Optional[Callable[[int], None]],
    ) -> List[Dict[str, Any]]:
        """
        Generate games using ThreadPoolExecutor (Phase 3).

        All worker threads share the same network and BatchedEvaluator instance,
        enabling true cross-worker batching with large batch sizes.
        """
        # Create thread pool if not already created
        if self.thread_pool is None:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            )

        # Distribute games evenly across workers
        games_per_worker = num_games // self.num_workers
        remaining_games = num_games % self.num_workers

        # Submit tasks to thread pool
        futures = []
        for worker_id in range(self.num_workers):
            # Give extra games to first few workers to handle remainder
            worker_games = games_per_worker + (1 if worker_id < remaining_games else 0)
            if worker_games > 0:
                future = self.thread_pool.submit(
                    _worker_generate_games_threaded,
                    worker_id,
                    worker_games,
                    num_players,
                    cards_to_deal,
                    self.network,
                    self.encoder,
                    self.masker,
                    self.num_determinizations,
                    self.simulations_per_determinization,
                    self.temperature_schedule,
                    self.batch_evaluator,  # Shared evaluator!
                    self.mcts_batch_size,
                    self.use_parallel_expansion,
                    self.parallel_batch_size,
                    self.enable_worker_profiling,
                    self.enable_worker_metrics,
                    self.run_id,
                    self.must_have_suit_bias,
                )
                futures.append(future)

        # Wait for all tasks to complete
        all_examples = []
        for future in concurrent.futures.as_completed(futures):
            worker_examples = future.result()
            all_examples.extend(worker_examples)

        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(num_games)

        return all_examples

    def generate_games_async(
        self,
        num_games: int,
        num_players: int = 4,
        cards_to_deal: int = 5,
    ) -> concurrent.futures.Future:
        """
        Generate games asynchronously (non-blocking).

        Returns immediately with a Future that will contain training examples
        when generation completes. Useful for running self-play in background
        while doing other work (e.g., training).

        Args:
            num_games: Total number of games to generate
            num_players: Players per game
            cards_to_deal: Cards to deal

        Returns:
            Future that will contain training examples when complete
        """
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Submit generation task to executor
        future = self.executor.submit(
            self.generate_games, num_games, num_players, cards_to_deal
        )

        return future

    def shutdown(self):
        """Shutdown the parallel workers and batch evaluator gracefully."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

        if self.batch_evaluator is not None:
            self.batch_evaluator.shutdown()
            self.batch_evaluator = None

        if self.gpu_server is not None:
            # Print final statistics before shutdown
            stats = self.gpu_server.get_stats()
            print(f"[SelfPlayEngine] GPU server statistics:")
            print(f"  Total requests: {stats.get('total_requests', 0)}")
            print(f"  Total batches: {stats.get('total_batches', 0)}")
            print(f"  Avg batch size: {stats.get('avg_batch_size', 0):.1f}")
            print(f"  Max batch size: {stats.get('max_batch_size', 0)}")

            self.gpu_server.shutdown()
            self.gpu_server = None
            self.gpu_clients.clear()


def _worker_generate_games_static(
    worker_id: int,
    num_games: int,
    num_players: int,
    cards_to_deal: int,
    network_state: Dict[str, torch.Tensor],
    num_determinizations: int,
    simulations_per_determinization: int,
    temperature_schedule: Optional[Callable[[int], float]],
    device: str = "cpu",
    use_batched_evaluator: bool = False,
    batch_size: int = 512,
    batch_timeout_ms: float = 10.0,
    mcts_batch_size: Optional[int] = None,
    use_parallel_expansion: bool = True,
    parallel_batch_size: int = 10,
    enable_worker_profiling: bool = False,
    enable_worker_metrics: bool = False,
    run_id: Optional[str] = None,
    must_have_suit_bias: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Static worker function for parallel game generation.

    This function is defined at module level (not as a method) so it can
    be pickled by multiprocessing. Each worker creates its own isolated
    SelfPlayWorker instance and generates games independently.

    Note:
        When use_batched_evaluator=True, each worker creates its own
        BatchedEvaluator instance. This provides intra-worker batching
        but not cross-worker batching (due to multiprocessing limitations).
        For full multi-game batching, consider using ThreadPoolExecutor
        instead of multiprocessing.Pool.

    Args:
        worker_id: Unique worker identifier
        num_games: Number of games this worker should generate
        num_players: Players per game
        cards_to_deal: Cards to deal
        network_state: Network weights (state dict)
        num_determinizations: Determinizations per MCTS search
        simulations_per_determinization: MCTS simulations per world
        temperature_schedule: Temperature schedule function
        device: Device to run network on ('cpu' or 'cuda')
        use_batched_evaluator: Create BatchedEvaluator for this worker
        batch_size: Maximum batch size for BatchedEvaluator
        batch_timeout_ms: Timeout in ms for batch collection
        mcts_batch_size: Phase 1 intra-game batching size (None for sequential)
        use_parallel_expansion: Use parallel MCTS expansion for better batching
        parallel_batch_size: Number of leaves to expand per iteration
        enable_worker_profiling: Enable cProfile profiling for worker 0 (default: False)

    Returns:
        List of training examples from all games
    """
    # Set random seed for this worker (ensures different games across workers)
    np.random.seed(worker_id + int(uuid.uuid4().int % 10000))
    torch.manual_seed(worker_id + int(uuid.uuid4().int % 10000))

    # Enable profiling for worker 0 if requested (useful for performance analysis)
    import cProfile
    profiler = cProfile.Profile() if (enable_worker_profiling and worker_id == 0) else None
    if profiler:
        profiler.enable()

    # Optional lightweight instrumentation
    if enable_worker_metrics:
        try:
            from ml.mcts import determinization as _det
            from ml.mcts import node as _node
            if hasattr(_det, 'reset_metrics') and hasattr(_det, 'enable_metrics'):
                _det.reset_metrics()
                _det.enable_metrics(True)
            if hasattr(_node, 'reset_metrics') and hasattr(_node, 'enable_metrics'):
                _node.reset_metrics()
                _node.enable_metrics(True)
        except Exception:
            pass

    # Create network instance for this worker
    # Infer network architecture from state dict
    state_dim = network_state["input_embedding.weight"].shape[1]
    embedding_dim = network_state["input_embedding.weight"].shape[0]

    # Count transformer layers
    num_layers = 0
    while f"transformer.layers.{num_layers}.self_attn.in_proj_weight" in network_state:
        num_layers += 1

    # Infer feedforward dimension from actual linear1 layer
    feedforward_dim = network_state["transformer.layers.0.linear1.weight"].shape[0]

    # Get other hyperparameters from state dict shapes
    num_heads = 8  # Default, could infer from attention weights if needed
    dropout = 0.1  # Default

    network = BlobNet(
        state_dim=state_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
    )
    network.load_state_dict(network_state)
    network.to(device)  # Move to specified device (CPU or GPU)

    # DIAGNOSTIC: Verify GPU setup on first worker only
    if worker_id == 0:
        actual_device = next(network.parameters()).device
        print(f"[Worker 0] Network on device: {actual_device} (requested: {device})")
        if device == "cuda" and torch.cuda.is_available():
            print(f"[Worker 0] GPU: {torch.cuda.get_device_name(0)}")

    network.eval()  # Set to evaluation mode

    # Create encoder and masker for this worker
    encoder = StateEncoder()
    masker = ActionMasker()

    # Create BatchedEvaluator if requested
    batch_evaluator = None
    if use_batched_evaluator:
        from ml.mcts.batch_evaluator import BatchedEvaluator

        batch_evaluator = BatchedEvaluator(
            network=network,
            max_batch_size=batch_size,
            timeout_ms=batch_timeout_ms,
            device=device,
            enable_profiling_metrics=enable_worker_metrics,
        )
        batch_evaluator.start()

    # Create SelfPlayWorker
    worker = SelfPlayWorker(
        network=network,
        encoder=encoder,
        masker=masker,
        num_determinizations=num_determinizations,
        simulations_per_determinization=simulations_per_determinization,
        temperature_schedule=temperature_schedule,
        use_imperfect_info=True,
        batch_evaluator=batch_evaluator,
        batch_size=mcts_batch_size,
        use_parallel_expansion=use_parallel_expansion,
        parallel_batch_size=parallel_batch_size,
        must_have_suit_bias=must_have_suit_bias,
    )

    # Generate games
    all_examples = []
    for game_idx in range(num_games):
        # Include UUID to ensure uniqueness across runs
        unique_id = str(uuid.uuid4())[:8]
        game_id = f"worker{worker_id}_game{game_idx}_{unique_id}"
        examples = worker.generate_game(
            num_players=num_players,
            cards_to_deal=cards_to_deal,
            game_id=game_id,
        )
        all_examples.extend(examples)

    # Cleanup batch evaluator
    if batch_evaluator is not None:
        batch_evaluator.shutdown()

    # Save worker profile to reveal actual bottlenecks (MCTS, network, game logic)
    if profiler:
        profiler.disable()
        profile_file = f"profile_worker{worker_id}.prof"
        profiler.dump_stats(profile_file)
        print(f"Worker {worker_id} profile saved to {profile_file}")

    # Save lightweight instrumentation metrics (per-worker JSON)
    if enable_worker_metrics:
        try:
            det_m = {}
            node_m = {}
            be_stats = None
            try:
                from ml.mcts import determinization as _det
                if hasattr(_det, 'get_metrics'):
                    det_m = _det.get_metrics()
            except Exception:
                pass
            try:
                from ml.mcts import node as _node
                if hasattr(_node, 'get_metrics'):
                    node_m = _node.get_metrics()
            except Exception:
                pass
            try:
                if use_batched_evaluator and batch_evaluator is not None:
                    be_stats = batch_evaluator.get_stats()
            except Exception:
                be_stats = None

            out = {
                'worker_id': worker_id,
                'run_id': run_id or 'unknown',
                'timestamp': int(_time.time()),
                'determinization': det_m,
                'node': node_m,
                'batch_evaluator': be_stats,
                'num_games': num_games,
                'device': device,
            }
            out_path = f"profile_{(run_id or 'run')}_worker{worker_id}_metrics.json"
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"Worker {worker_id} metrics saved to {out_path}")
        except Exception as _e:
            print(f"[Worker {worker_id}] Failed to write metrics: {_e}")

    return all_examples


def _worker_generate_games_with_gpu_server(
    worker_id: int,
    num_games: int,
    num_players: int,
    cards_to_deal: int,
    gpu_server_request_queue: mp.Queue,
    gpu_server_response_queue: mp.Queue,
    client_id: str,
    num_determinizations: int,
    simulations_per_determinization: int,
    temperature_schedule: Optional[Callable[[int], float]],
    use_parallel_expansion: bool = True,
    parallel_batch_size: int = 10,
    enable_worker_profiling: bool = False,
    enable_worker_metrics: bool = False,
    run_id: Optional[str] = None,
    must_have_suit_bias: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Worker function for GPU server mode (Phase 3.5).

    This worker does not create its own network. Instead, it receives a GPU
    server client that sends evaluation requests to the centralized GPU server
    process running in the main process.

    Args:
        worker_id: Unique worker identifier
        num_games: Number of games this worker should generate
        num_players: Players per game
        cards_to_deal: Cards to deal
        gpu_server_request_queue: Shared request queue to GPU server
        gpu_server_response_queue: Private response queue from GPU server
        client_id: Unique client identifier for this worker
        num_determinizations: Determinizations per MCTS search
        simulations_per_determinization: MCTS simulations per world
        temperature_schedule: Temperature schedule function
        use_parallel_expansion: Use parallel MCTS expansion for better batching
        parallel_batch_size: Number of leaves to expand per iteration
        enable_worker_profiling: Enable cProfile profiling for worker 0 (default: False)

    Returns:
        List of training examples from all games
    """
    # Set random seed for this worker
    np.random.seed(worker_id + int(uuid.uuid4().int % 10000))
    torch.manual_seed(worker_id + int(uuid.uuid4().int % 10000))

    # Enable profiling for worker 0 if requested (useful for performance analysis)
    import cProfile
    profiler = cProfile.Profile() if (enable_worker_profiling and worker_id == 0) else None
    if profiler:
        profiler.enable()

    # Create GPU server client
    from ml.mcts.gpu_server import GPUServerClient

    gpu_client = GPUServerClient(
        request_queue=gpu_server_request_queue,
        response_queue=gpu_server_response_queue,
        client_id=client_id,
    )

    # Create encoder and masker for this worker
    encoder = StateEncoder()
    masker = ActionMasker()

    # Create SelfPlayWorker (no network needed, using GPU server)
    worker = SelfPlayWorker(
        network=None,  # Not used when gpu_server_client is provided
        encoder=encoder,
        masker=masker,
        num_determinizations=num_determinizations,
        simulations_per_determinization=simulations_per_determinization,
        temperature_schedule=temperature_schedule,
        use_imperfect_info=True,
        batch_evaluator=None,
        gpu_server_client=gpu_client,
        use_parallel_expansion=use_parallel_expansion,
        parallel_batch_size=parallel_batch_size,
        must_have_suit_bias=must_have_suit_bias,
    )

    # Generate games
    all_examples = []
    for game_idx in range(num_games):
        unique_id = str(uuid.uuid4())[:8]
        game_id = f"worker{worker_id}_game{game_idx}_{unique_id}"
        examples = worker.generate_game(
            num_players=num_players,
            cards_to_deal=cards_to_deal,
            game_id=game_id,
        )
        all_examples.extend(examples)

    # Save worker profile to reveal actual bottlenecks (MCTS, network, IPC)
    if profiler:
        profiler.disable()
        profile_file = f"profile_worker{worker_id}_gpu_server.prof"
        profiler.dump_stats(profile_file)
        print(f"Worker {worker_id} (GPU server mode) profile saved to {profile_file}")

    # Save lightweight instrumentation metrics (per-worker JSON)
    if enable_worker_metrics:
        try:
            det_m = {}
            node_m = {}
            try:
                from ml.mcts import determinization as _det
                if hasattr(_det, 'get_metrics'):
                    det_m = _det.get_metrics()
            except Exception:
                pass
            try:
                from ml.mcts import node as _node
                if hasattr(_node, 'get_metrics'):
                    node_m = _node.get_metrics()
            except Exception:
                pass
            out = {
                'worker_id': worker_id,
                'run_id': run_id or 'unknown',
                'timestamp': int(_time.time()),
                'determinization': det_m,
                'node': node_m,
                'batch_evaluator': None,
                'num_games': num_games,
                'device': 'cuda',
            }
            out_path = f"profile_{(run_id or 'run')}_worker{worker_id}_metrics.json"
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"Worker {worker_id} metrics saved to {out_path}")
        except Exception as _e:
            print(f"[Worker {worker_id}] Failed to write metrics: {_e}")

    return all_examples


def _worker_generate_games_threaded(
    worker_id: int,
    num_games: int,
    num_players: int,
    cards_to_deal: int,
    network: BlobNet,
    encoder: StateEncoder,
    masker: ActionMasker,
    num_determinizations: int,
    simulations_per_determinization: int,
    temperature_schedule: Optional[Callable[[int], float]],
    batch_evaluator: Optional["BatchedEvaluator"],
    mcts_batch_size: Optional[int] = None,
    use_parallel_expansion: bool = True,
    parallel_batch_size: int = 10,
    enable_worker_profiling: bool = False,
    enable_worker_metrics: bool = False,
    run_id: Optional[str] = None,
    must_have_suit_bias: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Threaded worker function for parallel game generation (Phase 3).

    This worker runs in a thread and shares the network and BatchedEvaluator
    with all other worker threads. This enables true cross-worker batching
    with large batch sizes (128-512+) for maximum GPU utilization.

    Key Differences from _worker_generate_games_static:
        - No network creation (shares main thread's network)
        - No BatchedEvaluator creation (shares main thread's evaluator)
        - Lighter weight (no pickling, no process overhead)
        - Enables cross-worker batching (all threads → same evaluator)

    Args:
        worker_id: Unique worker identifier
        num_games: Number of games this worker should generate
        num_players: Players per game
        cards_to_deal: Cards to deal
        network: Shared neural network instance (thread-safe during eval)
        encoder: StateEncoder instance (thread-safe)
        masker: ActionMasker instance (thread-safe)
        num_determinizations: Determinizations per MCTS search
        simulations_per_determinization: MCTS simulations per world
        temperature_schedule: Temperature schedule function
        batch_evaluator: Shared BatchedEvaluator (thread-safe)
        mcts_batch_size: Phase 1 intra-game batching size (None for sequential)
        use_parallel_expansion: Use parallel MCTS expansion for better batching
        parallel_batch_size: Number of leaves to expand per iteration
        enable_worker_profiling: Enable cProfile profiling for worker 0 (default: False)

    Returns:
        List of training examples from all games
    """
    # Set random seed for this worker (ensures different games across workers)
    np.random.seed(worker_id + int(uuid.uuid4().int % 10000))
    torch.manual_seed(worker_id + int(uuid.uuid4().int % 10000))

    # Enable profiling for worker 0 if requested (useful for performance analysis)
    import cProfile
    profiler = cProfile.Profile() if (enable_worker_profiling and worker_id == 0) else None
    if profiler:
        profiler.enable()

    # Optional lightweight instrumentation (threaded mode)
    if enable_worker_metrics:
        try:
            from ml.mcts import determinization as _det
            from ml.mcts import node as _node
            if hasattr(_det, 'reset_metrics') and hasattr(_det, 'enable_metrics'):
                _det.reset_metrics()
                _det.enable_metrics(True)
            if hasattr(_node, 'reset_metrics') and hasattr(_node, 'enable_metrics'):
                _node.reset_metrics()
                _node.enable_metrics(True)
        except Exception:
            pass

    # Create SelfPlayWorker with shared network and evaluator
    worker = SelfPlayWorker(
        network=network,  # Shared network (no copy needed)
        encoder=encoder,  # Shared encoder
        masker=masker,  # Shared masker
        num_determinizations=num_determinizations,
        simulations_per_determinization=simulations_per_determinization,
        temperature_schedule=temperature_schedule,
        use_parallel_expansion=use_parallel_expansion,
        parallel_batch_size=parallel_batch_size,
        use_imperfect_info=True,
        batch_evaluator=batch_evaluator,  # Shared evaluator!
        batch_size=mcts_batch_size,
        must_have_suit_bias=must_have_suit_bias,
    )

    # Generate games
    all_examples = []
    for game_idx in range(num_games):
        # Include UUID to ensure uniqueness across runs
        unique_id = str(uuid.uuid4())[:8]
        game_id = f"worker{worker_id}_game{game_idx}_{unique_id}"
        examples = worker.generate_game(
            num_players=num_players,
            cards_to_deal=cards_to_deal,
            game_id=game_id,
        )
        all_examples.extend(examples)

    # No cleanup needed - evaluator is shared and managed by main thread

    # Save worker profile to reveal actual bottlenecks (MCTS, network, batching)
    if profiler:
        profiler.disable()
        profile_file = f"profile_worker{worker_id}_threaded.prof"
        profiler.dump_stats(profile_file)
        print(f"Worker {worker_id} (threaded mode) profile saved to {profile_file}")

    # Save lightweight instrumentation metrics (per-thread JSON)
    if enable_worker_metrics:
        try:
            det_m = {}
            node_m = {}
            try:
                from ml.mcts import determinization as _det
                if hasattr(_det, 'get_metrics'):
                    det_m = _det.get_metrics()
            except Exception:
                pass
            try:
                from ml.mcts import node as _node
                if hasattr(_node, 'get_metrics'):
                    node_m = _node.get_metrics()
            except Exception:
                pass
            out = {
                'worker_id': worker_id,
                'run_id': run_id or 'unknown',
                'timestamp': int(_time.time()),
                'determinization': det_m,
                'node': node_m,
                'batch_evaluator': None,  # shared, summarized by main thread
                'num_games': num_games,
                'device': 'cuda',
            }
            out_path = f"profile_{(run_id or 'run')}_thread{worker_id}_metrics.json"
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"Thread worker {worker_id} metrics saved to {out_path}")
        except Exception as _e:
            print(f"[Thread worker {worker_id}] Failed to write metrics: {_e}")

    return all_examples
    # Optional lightweight instrumentation
    if enable_worker_metrics:
        try:
            from ml.mcts import determinization as _det
            from ml.mcts import node as _node
            if hasattr(_det, 'reset_metrics') and hasattr(_det, 'enable_metrics'):
                _det.reset_metrics()
                _det.enable_metrics(True)
            if hasattr(_node, 'reset_metrics') and hasattr(_node, 'enable_metrics'):
                _node.reset_metrics()
                _node.enable_metrics(True)
        except Exception:
            pass
