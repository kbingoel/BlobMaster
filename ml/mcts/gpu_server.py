"""
GPU Inference Server for Multiprocessing-Based Self-Play

This module implements a centralized GPU inference server that receives evaluation
requests from multiple MCTS worker processes, batches them together, and performs
single GPU forward passes for maximum efficiency.

Problem Statement:
    - Phase 3 (threading) failed on Windows due to GIL contention (6.9-9.6 games/min)
    - Multiprocessing avoids GIL but creates 32 separate CUDA contexts
    - Need: True parallelism (multiprocessing) + centralized GPU batching

Solution Architecture:
    - Single GPU server process owns the neural network
    - 32 MCTS worker processes send requests via multiprocessing.Queue
    - Server accumulates requests until batch full or timeout
    - Server performs batched GPU inference and returns results
    - Workers continue MCTS with results

Expected Performance:
    - Batch sizes: 128-320 avg (vs 3.5 currently)
    - Throughput: 800-1,100 games/min (vs 68.3 currently)
    - GPU utilization: 70-85% (vs 15% currently)
    - Training time: 4-7 days (vs 50 days currently)

Example Usage:
    >>> # Main process
    >>> server = GPUInferenceServer(network, device='cuda', max_batch_size=512)
    >>> server.start()
    >>>
    >>> # Worker processes
    >>> client = server.create_client()
    >>> policy, value = client.evaluate(state_tensor, legal_mask)
    >>>
    >>> # Cleanup
    >>> server.shutdown()
"""

import torch
import multiprocessing as mp
import time
import queue
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import uuid

from ml.network.model import BlobNet


@dataclass
class InferenceRequest:
    """
    Request for neural network evaluation (picklable for multiprocessing).

    Attributes:
        request_id: Unique identifier for this request
        state: Encoded game state tensor (256-dim)
        mask: Legal action mask tensor (52-dim)
        response_queue_id: ID of the queue to send result to
    """
    request_id: str
    state: torch.Tensor
    mask: torch.Tensor
    response_queue_id: str


@dataclass
class InferenceResult:
    """
    Result from neural network evaluation.

    Attributes:
        request_id: Matches the request this result is for
        policy: Action probabilities (52-dim)
        value: Value estimate (scalar)
        error: Optional error message if evaluation failed
    """
    request_id: str
    policy: torch.Tensor
    value: torch.Tensor
    error: Optional[str] = None


class GPUServerClient:
    """
    Client handle for workers to communicate with GPU server.

    Each worker process gets its own client with a unique response queue.
    The client sends requests to the shared request queue and waits for
    results on its private response queue.
    """

    def __init__(
        self,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        client_id: str,
    ):
        """
        Initialize GPU server client.

        Args:
            request_queue: Shared queue for sending requests to server
            response_queue: Private queue for receiving results from server
            client_id: Unique identifier for this client
        """
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.client_id = client_id

    def evaluate(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
        timeout: float = 30.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Request neural network evaluation from GPU server.

        Sends request to server and blocks until result is received.

        Args:
            state: Encoded game state tensor (256-dim)
            mask: Legal action mask tensor (52-dim)
            timeout: Maximum time to wait for result (seconds)

        Returns:
            Tuple of (policy, value) tensors

        Raises:
            TimeoutError: If result not received within timeout
            RuntimeError: If server returns an error
        """
        # Generate unique request ID
        request_id = f"{self.client_id}_{uuid.uuid4().hex[:8]}"

        # Create request
        request = InferenceRequest(
            request_id=request_id,
            state=state.cpu(),  # Move to CPU for pickling
            mask=mask.cpu(),
            response_queue_id=self.client_id,
        )

        # Send request to server
        self.request_queue.put(request)

        # Wait for result
        try:
            result = self.response_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"GPU server did not respond within {timeout}s for request {request_id}"
            )

        # Check for errors
        if result.error is not None:
            raise RuntimeError(f"GPU server error: {result.error}")

        # Verify request ID matches
        if result.request_id != request_id:
            raise RuntimeError(
                f"Request ID mismatch: expected {request_id}, got {result.request_id}"
            )

        return result.policy, result.value


class GPUInferenceServer:
    """
    Centralized GPU inference server for multiprocessing-based self-play.

    Runs in a separate process and handles all neural network evaluations
    for multiple MCTS worker processes. Batches requests for maximum GPU
    efficiency while maintaining low latency.

    Performance Characteristics:
        - Batch size: 128-512 (depends on worker load)
        - Latency: 10-20ms (batch timeout + inference time)
        - Throughput: Limited by GPU, not by Python overhead
        - GPU utilization: 70-85% (much higher than Phase 3's 2-8%)

    Attributes:
        network: Neural network for evaluation
        device: Device to run network on ('cuda')
        max_batch_size: Maximum batch size for GPU
        timeout_ms: Timeout in milliseconds for batch collection
        running: Whether server is running
        process: Background server process
        request_queue: Queue for receiving requests from workers
        response_queues: Dict of queues for sending results back (by client_id)
    """

    def __init__(
        self,
        network: BlobNet,
        device: str = "cuda",
        max_batch_size: int = 512,
        timeout_ms: float = 10.0,
    ):
        """
        Initialize GPU inference server.

        Args:
            network: Neural network for evaluation
            device: Device to run network on (default: 'cuda')
            max_batch_size: Maximum batch size (default: 512)
                          Higher = better GPU utilization but more latency
            timeout_ms: Batch collection timeout in milliseconds (default: 10ms)
                       Lower = lower latency but smaller batches
                       Higher = larger batches but more latency
        """
        self.network = network
        self.device = device
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms

        # Get network state dict for transferring to server process
        self.network_state = network.state_dict()
        self.network_config = {
            'state_dim': network.state_dim,
            'embedding_dim': network.embedding_dim,
            'num_layers': network.num_layers,
            'num_heads': network.num_heads,
            'feedforward_dim': network.feedforward_dim,
            'dropout': network.dropout,
            'max_bid': network.max_bid,
            'max_cards': network.max_cards,
        }

        # Process management
        self.running = False
        self.process: Optional[mp.Process] = None

        # Communication queues (initialized in start())
        self.request_queue: Optional[mp.Queue] = None
        self.response_queues = None  # Will be manager.dict() in start()
        self.manager = None  # Will be mp.Manager() in start()

        # Shutdown event
        self.shutdown_event: Optional[mp.Event] = None

        # Statistics tracking (shared memory)
        self.stats_dict: Optional[Dict[str, Any]] = None

    def start(self):
        """
        Start the GPU inference server process.

        Creates communication queues and spawns the server process.
        """
        if self.running:
            raise RuntimeError("GPU server is already running")

        # Create manager for cross-process communication (Windows-compatible)
        self.manager = mp.Manager()

        # Create communication primitives using manager for Windows compatibility
        self.request_queue = self.manager.Queue(maxsize=1000)  # Buffer up to 1000 requests
        self.shutdown_event = mp.Event()

        # Use manager dict for response queues so updates are visible to server process
        self.response_queues = self.manager.dict()

        # Create shared dict for statistics
        self.stats_dict = self.manager.dict({
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'max_batch_size': 0,
        })

        # Start server process
        self.process = mp.Process(
            target=_server_process_loop,
            args=(
                self.request_queue,
                self.response_queues,
                self.shutdown_event,
                self.network_state,
                self.network_config,
                self.device,
                self.max_batch_size,
                self.timeout_ms,
                self.stats_dict,
            ),
            daemon=True,  # Automatically terminate when main process exits
        )
        self.process.start()
        self.running = True

        # Give server time to initialize
        time.sleep(0.5)

    def create_client(self, client_id: Optional[str] = None) -> GPUServerClient:
        """
        Create a client handle for a worker process.

        Each worker should get its own client with a unique response queue.

        Args:
            client_id: Unique identifier for this client (auto-generated if None)

        Returns:
            GPUServerClient instance for the worker
        """
        if not self.running:
            raise RuntimeError("GPU server is not running")

        # Generate client ID if not provided
        if client_id is None:
            client_id = f"client_{uuid.uuid4().hex[:8]}"

        # Create response queue for this client using manager (Windows-compatible)
        response_queue = self.manager.Queue()
        self.response_queues[client_id] = response_queue

        # Create and return client
        return GPUServerClient(
            request_queue=self.request_queue,
            response_queue=response_queue,
            client_id=client_id,
        )

    def shutdown(self, timeout: float = 5.0):
        """
        Shutdown the GPU inference server.

        Signals the server to stop and waits for it to terminate.

        Args:
            timeout: Maximum time to wait for shutdown (seconds)
        """
        if not self.running:
            return

        # Signal shutdown
        self.shutdown_event.set()

        # Wait for process to terminate
        if self.process is not None:
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                # Force terminate if it doesn't shut down gracefully
                self.process.terminate()
                self.process.join(timeout=1.0)

        self.running = False
        self.process = None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.

        Returns:
            Dictionary with statistics (total requests, avg batch size, etc.)
        """
        if self.stats_dict is None:
            return {}
        return dict(self.stats_dict)


def _server_process_loop(
    request_queue: mp.Queue,
    response_queues: Dict[str, mp.Queue],
    shutdown_event: mp.Event,
    network_state: Dict[str, torch.Tensor],
    network_config: Dict[str, int],
    device: str,
    max_batch_size: int,
    timeout_ms: float,
    stats_dict: Dict[str, Any],
):
    """
    Main loop for GPU inference server process.

    This function runs in a separate process and handles all GPU evaluations.
    It accumulates requests into batches and performs batched inference.

    Args:
        request_queue: Queue for receiving requests from workers
        response_queues: Dict of queues for sending results back
        shutdown_event: Event to signal shutdown
        network_state: State dict for neural network
        network_config: Config dict for neural network
        device: Device to run on ('cuda')
        max_batch_size: Maximum batch size
        timeout_ms: Batch collection timeout in milliseconds
        stats_dict: Shared dict for statistics
    """
    # Initialize network in this process
    network = BlobNet(**network_config)
    network.load_state_dict(network_state)
    network.to(device)
    network.eval()

    # Convert timeout to seconds
    timeout_sec = timeout_ms / 1000.0

    # Statistics
    total_requests = 0
    total_batches = 0
    total_batch_size_sum = 0
    max_batch_size_seen = 0

    print(f"[GPU Server] Started on {device} (max_batch={max_batch_size}, timeout={timeout_ms}ms)")

    # Main server loop
    while not shutdown_event.is_set():
        # Accumulate requests
        batch_requests = []
        batch_start_time = time.time()

        # Collect requests until batch full or timeout
        while len(batch_requests) < max_batch_size:
            remaining_time = timeout_sec - (time.time() - batch_start_time)
            if remaining_time <= 0:
                break

            try:
                request = request_queue.get(timeout=remaining_time)
                batch_requests.append(request)
            except queue.Empty:
                # Timeout reached
                break

        # Skip if no requests
        if len(batch_requests) == 0:
            continue

        # Batch inference
        try:
            # Stack tensors into batch
            states = torch.stack([req.state for req in batch_requests]).to(device)
            masks = torch.stack([req.mask for req in batch_requests]).to(device)

            # Run neural network
            with torch.no_grad():
                policies, values = network(states, masks)

            # Distribute results back to workers
            for i, request in enumerate(batch_requests):
                result = InferenceResult(
                    request_id=request.request_id,
                    policy=policies[i].cpu(),
                    value=values[i].cpu(),
                    error=None,
                )

                # Send result to appropriate response queue
                response_queue = response_queues.get(request.response_queue_id)
                if response_queue is not None:
                    response_queue.put(result)
                else:
                    print(f"[GPU Server] Warning: No response queue for client {request.response_queue_id}")

            # Update statistics
            batch_size = len(batch_requests)
            total_requests += batch_size
            total_batches += 1
            total_batch_size_sum += batch_size
            max_batch_size_seen = max(max_batch_size_seen, batch_size)

            # Update shared stats dict periodically
            if total_batches % 10 == 0:
                stats_dict['total_requests'] = total_requests
                stats_dict['total_batches'] = total_batches
                stats_dict['avg_batch_size'] = total_batch_size_sum / total_batches
                stats_dict['max_batch_size'] = max_batch_size_seen

        except Exception as e:
            # Send error results to all requesters
            error_msg = f"Server error: {str(e)}"
            for request in batch_requests:
                result = InferenceResult(
                    request_id=request.request_id,
                    policy=torch.zeros(52),
                    value=torch.tensor(0.0),
                    error=error_msg,
                )

                response_queue = response_queues.get(request.response_queue_id)
                if response_queue is not None:
                    response_queue.put(result)

            print(f"[GPU Server] Error processing batch: {e}")

    # Final statistics update
    if total_batches > 0:
        stats_dict['total_requests'] = total_requests
        stats_dict['total_batches'] = total_batches
        stats_dict['avg_batch_size'] = total_batch_size_sum / total_batches
        stats_dict['max_batch_size'] = max_batch_size_seen

        print(f"[GPU Server] Shutting down. Stats:")
        print(f"  Total requests: {total_requests}")
        print(f"  Total batches: {total_batches}")
        print(f"  Avg batch size: {total_batch_size_sum / total_batches:.1f}")
        print(f"  Max batch size: {max_batch_size_seen}")
