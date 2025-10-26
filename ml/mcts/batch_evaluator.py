"""
Batched Neural Network Evaluator for Multi-Game MCTS

This module implements a centralized batching service that collects neural network
evaluation requests from multiple MCTS instances and batches them into single GPU
calls for improved performance.

Problem:
    - 16 parallel self-play workers each running MCTS
    - Each MCTS makes 90 network calls per move (even with Phase 1 batching)
    - Sequential calls result in low GPU utilization (~1%)

Solution:
    - Centralized BatchedEvaluator service
    - Collects requests from all MCTS instances
    - Batches them into single GPU call (512-2048 samples)
    - Expected speedup: 50-100x total

Architecture:
    - Background thread runs evaluation loop
    - Thread-safe queue for request collection
    - Blocking API: MCTS waits for result via Queue.get()
    - Timeout mechanism to prevent starvation with small batches

Example:
    >>> evaluator = BatchedEvaluator(network, max_batch_size=512, timeout_ms=10)
    >>> evaluator.start()
    >>>
    >>> # In MCTS (called from multiple threads)
    >>> policy, value = evaluator.evaluate(state_tensor, legal_mask)
    >>>
    >>> evaluator.shutdown()
"""

import torch
import threading
import queue
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass

from ml.network.model import BlobNet


@dataclass
class EvaluationRequest:
    """
    Request for neural network evaluation.

    Attributes:
        state: Encoded game state tensor (256-dim)
        mask: Legal action mask tensor (52-dim)
        result_queue: Queue to put result into when evaluation completes
        request_id: Unique identifier for debugging
    """

    state: torch.Tensor
    mask: torch.Tensor
    result_queue: queue.Queue
    request_id: int


@dataclass
class EvaluationResult:
    """
    Result from neural network evaluation.

    Attributes:
        policy: Action probabilities (52-dim)
        value: Value estimate (scalar)
        error: Optional error message if evaluation failed
    """

    policy: torch.Tensor
    value: torch.Tensor
    error: Optional[str] = None


class BatchedEvaluator:
    """
    Centralized batched neural network evaluator.

    Collects evaluation requests from multiple MCTS instances and batches
    them into single GPU calls for improved performance.

    This service runs a background thread that:
    1. Collects requests from a thread-safe queue
    2. Accumulates requests until batch is full or timeout expires
    3. Performs single batched neural network forward pass
    4. Distributes results back to requesters via their result queues

    Performance Impact:
        - Without batching: 16 workers × 90 calls/move = 1440 sequential calls
        - With batching: 1-5 batched calls (batch_size=512-2048)
        - Expected speedup: 50-100x

    Attributes:
        network: Neural network for evaluation
        max_batch_size: Maximum batch size (default: 512)
        timeout_ms: Timeout in milliseconds for batch collection (default: 10ms)
        device: Device to run network on ('cpu' or 'cuda')
        running: Whether background thread is running
        thread: Background evaluation thread
        request_queue: Thread-safe queue for incoming requests
        next_request_id: Counter for assigning request IDs
    """

    def __init__(
        self,
        network: BlobNet,
        max_batch_size: int = 512,
        timeout_ms: float = 10.0,
        device: Optional[str] = None,
    ):
        """
        Initialize batched evaluator.

        Args:
            network: Neural network for evaluation (BlobNet instance)
            max_batch_size: Maximum number of requests to batch (default: 512)
                           Larger = better GPU utilization but more latency
                           Recommended: 512 for training, 256 for inference
            timeout_ms: Timeout in milliseconds for batch collection (default: 10ms)
                       Lower = lower latency but smaller batches
                       Higher = larger batches but more latency
            device: Device to run network on ('cpu' or 'cuda')
                   If None, infers from network parameters
        """
        self.network = network
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms / 1000.0  # Convert to seconds

        # Detect device from network if not specified
        if device is None:
            self.device = next(network.parameters()).device
        else:
            self.device = torch.device(device)

        # Ensure network is on correct device and in eval mode
        self.network.to(self.device)
        self.network.eval()

        # Background thread management
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Thread-safe request queue (unbounded)
        self.request_queue: queue.Queue[EvaluationRequest] = queue.Queue()

        # Request ID counter for debugging
        self.next_request_id = 0
        self.request_id_lock = threading.Lock()

        # Statistics for monitoring
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0
        self.stats_lock = threading.Lock()

    def start(self):
        """
        Start the background evaluation thread.

        Call this before using evaluate(). The thread will run until
        shutdown() is called.

        Example:
            >>> evaluator = BatchedEvaluator(network)
            >>> evaluator.start()
            >>> # Now can call evaluate() from multiple threads
        """
        if self.running:
            return  # Already running

        self.running = True
        self.thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.thread.start()

    def shutdown(self, timeout: float = 5.0):
        """
        Shutdown the background evaluation thread.

        Waits for thread to finish processing any pending requests.

        Args:
            timeout: Maximum time to wait for thread to finish (seconds)

        Example:
            >>> evaluator.shutdown()
        """
        if not self.running:
            return  # Already stopped

        self.running = False

        # Wait for thread to finish
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            self.thread = None

    def evaluate(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate a game state using batched neural network inference.

        This method is blocking - it waits for the background thread to
        process the request and return the result. Multiple threads can
        call this method concurrently.

        Args:
            state: Encoded game state tensor (256-dim)
            mask: Legal action mask tensor (52-dim)

        Returns:
            Tuple of (policy, value):
                - policy: Action probabilities (52-dim tensor)
                - value: Value estimate (scalar tensor)

        Raises:
            RuntimeError: If evaluator is not running or evaluation fails

        Example:
            >>> state = encoder.encode(game, player)
            >>> mask = masker.create_mask(...)
            >>> policy, value = evaluator.evaluate(state, mask)
        """
        if not self.running:
            raise RuntimeError(
                "BatchedEvaluator is not running. Call start() first."
            )

        # Create result queue for this request
        result_queue: queue.Queue[EvaluationResult] = queue.Queue(maxsize=1)

        # Get unique request ID
        with self.request_id_lock:
            request_id = self.next_request_id
            self.next_request_id += 1

        # Create and submit request
        request = EvaluationRequest(
            state=state,
            mask=mask,
            result_queue=result_queue,
            request_id=request_id,
        )

        self.request_queue.put(request)

        # Wait for result (blocking)
        result = result_queue.get()

        # Check for errors
        if result.error is not None:
            raise RuntimeError(f"Evaluation failed: {result.error}")

        return result.policy, result.value

    def _evaluation_loop(self):
        """
        Background thread that processes evaluation requests in batches.

        This loop runs continuously until shutdown() is called. It:
        1. Collects requests from queue until batch is full or timeout
        2. Performs batched neural network forward pass
        3. Distributes results back to requesters

        The loop uses a timeout mechanism to ensure low latency even when
        the batch doesn't fill up (e.g., at end of training).
        """
        while self.running:
            # Collect batch of requests
            batch = self._collect_batch()

            if not batch:
                # No requests collected, sleep briefly and retry
                time.sleep(0.001)  # 1ms
                continue

            # Process batch
            self._process_batch(batch)

    def _collect_batch(self) -> List[EvaluationRequest]:
        """
        Collect a batch of evaluation requests.

        Collects requests from the queue until either:
        1. Batch is full (reached max_batch_size)
        2. Timeout expires (no new requests for timeout_ms)

        Returns:
            List of evaluation requests (may be empty if no requests available)
        """
        batch: List[EvaluationRequest] = []
        deadline = time.time() + self.timeout_ms

        while len(batch) < self.max_batch_size:
            # Calculate remaining time until deadline
            remaining_time = deadline - time.time()

            if remaining_time <= 0 and batch:
                # Timeout expired and we have at least one request
                break

            try:
                # Try to get request with timeout
                timeout = max(0.001, remaining_time)  # At least 1ms
                request = self.request_queue.get(timeout=timeout)
                batch.append(request)
            except queue.Empty:
                # No request available within timeout
                if batch:
                    # We have some requests, process them
                    break
                else:
                    # No requests at all, return empty batch
                    return []

        return batch

    def _process_batch(self, batch: List[EvaluationRequest]):
        """
        Process a batch of evaluation requests.

        Performs batched neural network inference and distributes results
        back to requesters via their result queues.

        Args:
            batch: List of evaluation requests to process
        """
        if not batch:
            return

        try:
            # Extract states and masks from requests
            states = [req.state for req in batch]
            masks = [req.mask for req in batch]

            # Stack into batch tensors and move to device
            state_batch = torch.stack(states).to(self.device)
            mask_batch = torch.stack(masks).to(self.device)

            # Batched neural network inference (single GPU call!)
            with torch.no_grad():
                policy_batch, value_batch = self.network(state_batch, mask_batch)

            # Distribute results back to requesters
            for i, request in enumerate(batch):
                policy = policy_batch[i].cpu()  # Move back to CPU
                value = value_batch[i].cpu()

                result = EvaluationResult(policy=policy, value=value)
                request.result_queue.put(result)

            # Update statistics
            with self.stats_lock:
                self.total_requests += len(batch)
                self.total_batches += 1
                self.total_batch_size += len(batch)

        except Exception as e:
            # Send error to all requesters
            error_msg = f"Batch evaluation failed: {str(e)}"
            for request in batch:
                result = EvaluationResult(
                    policy=torch.zeros(52),
                    value=torch.tensor(0.0),
                    error=error_msg,
                )
                request.result_queue.put(result)

    def get_stats(self) -> dict:
        """
        Get evaluation statistics for monitoring.

        Returns:
            Dictionary with keys:
                - total_requests: Total number of requests processed
                - total_batches: Total number of batches processed
                - avg_batch_size: Average batch size
                - queue_size: Current number of pending requests

        Example:
            >>> stats = evaluator.get_stats()
            >>> print(f"Average batch size: {stats['avg_batch_size']:.1f}")
            >>> print(f"Pending requests: {stats['queue_size']}")
        """
        with self.stats_lock:
            avg_batch_size = (
                self.total_batch_size / self.total_batches
                if self.total_batches > 0
                else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "total_batches": self.total_batches,
                "avg_batch_size": avg_batch_size,
                "queue_size": self.request_queue.qsize(),
            }

    def reset_stats(self):
        """Reset statistics counters."""
        with self.stats_lock:
            self.total_requests = 0
            self.total_batches = 0
            self.total_batch_size = 0

    def __enter__(self):
        """Context manager entry - start the evaluator."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shutdown the evaluator."""
        self.shutdown()
