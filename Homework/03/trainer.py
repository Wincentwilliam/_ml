"""
trainer.py - Training loop wrapper with callback support for SSE streaming.
Decouples the training logic from FastAPI while providing real-time updates.
"""

import threading
import time
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass, field
import queue

from nn0 import (
    GridPulseNet, Value, get_optimizer, mse_loss,
    create_training_data, TrainingState, Optimizer
)


@dataclass
class TrainingUpdate:
    """A single training update to be sent via SSE."""
    epoch: int
    loss: float
    gradient_norms: Dict[str, float]
    learning_rate: float
    optimizer: str
    is_complete: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'epoch': self.epoch,
            'loss': self.loss,
            'gradient_norms': self.gradient_norms,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'is_complete': self.is_complete,
            'error': self.error
        }


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    learning_rate: float = 0.01
    optimizer: str = "adam"
    epochs: int = 100
    batch_size: int = 16
    update_interval: int = 5  # Send update every N epochs


class TrainingRunner:
    """
    Manages training in a background thread with callback support.
    Thread-safe state management for start/stop/reset operations.
    """

    def __init__(self, update_callback: Optional[Callable[[TrainingUpdate], None]] = None):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._training_thread: Optional[threading.Thread] = None
        self._update_callback = update_callback

        # Model and training state - initialize model at creation
        self._model: GridPulseNet = GridPulseNet()
        self._optimizer: Optional[Optimizer] = None
        self._training_data: list = []
        self._config: Optional[TrainingConfig] = None
        self._state = TrainingState()

        # Update queue for SSE (decouples training from HTTP)
        self._update_queue: queue.Queue = queue.Queue()

    def reset(self):
        """Reset the model and training state."""
        with self._lock:
            self.stop()
            self._model = GridPulseNet()
            self._state = TrainingState()
            self._update_queue = queue.Queue()

    def start(self, config: TrainingConfig):
        """Start training in a background thread."""
        with self._lock:
            if self._training_thread and self._training_thread.is_alive():
                return  # Already training

            self._stop_event.clear()
            self._config = config

            # Initialize model if needed
            if self._model is None:
                self._model = GridPulseNet()

            # Initialize optimizer
            params = self._model.parameters()
            self._optimizer = get_optimizer(config.optimizer, params, config.learning_rate)

            # Generate training data
            self._training_data = create_training_data(200)

            # Update state
            self._state.is_training = True
            self._state.total_epochs = config.epochs
            self._state.learning_rate = config.learning_rate
            self._state.optimizer_name = config.optimizer

            # Start training thread
            self._training_thread = threading.Thread(target=self._training_loop, daemon=True)
            self._training_thread.start()

    def stop(self):
        """Signal training to stop."""
        self._stop_event.set()
        with self._lock:
            self._state.is_training = False

    def is_training(self) -> bool:
        """Check if training is currently active."""
        return self._state.is_training

    def get_state(self) -> TrainingState:
        """Get current training state."""
        return self._state

    def get_state_dict(self) -> dict:
        """Get serializable state dict."""
        return self._state.to_dict()

    def get_model_state(self) -> dict:
        """Get serializable model state."""
        if self._model is None:
            return {}
        return self._model.get_state()

    def get_gradient_norms(self) -> dict:
        """Get current gradient norms."""
        if self._model is None:
            return {}
        return self._model.get_gradient_norms()

    def poll_update(self, timeout: float = 0.1) -> Optional[TrainingUpdate]:
        """Poll for the next training update (non-blocking)."""
        try:
            return self._update_queue.get_nowait()
        except queue.Empty:
            return None

    def _training_loop(self):
        """Main training loop running in background thread."""
        try:
            for epoch in range(self._config.epochs):
                if self._stop_event.is_set():
                    break

                epoch_loss = 0.0

                # Mini-batch training
                batch_size = min(self._config.batch_size, len(self._training_data))
                indices = list(range(len(self._training_data)))

                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_loss = 0.0

                    # Zero gradients
                    self._model.zero_grad()

                    # Forward pass and accumulate loss
                    for idx in batch_indices:
                        inputs, target = self._training_data[idx]
                        x = [Value(inp) for inp in inputs]
                        pred = self._model(x)
                        loss = mse_loss(pred, Value(target))
                        batch_loss += loss.data

                    # Backward pass
                    avg_loss = batch_loss / len(batch_indices)
                    avg_loss_value = Value(avg_loss)
                    avg_loss_value.backward()

                    # Optimizer step
                    self._optimizer.step()

                    epoch_loss += batch_loss

                # Average loss for epoch
                epoch_loss /= len(self._training_data)

                # Update state
                with self._lock:
                    self._state.epoch = epoch + 1
                    self._state.current_loss = epoch_loss
                    self._state.history.append({
                        'epoch': epoch + 1,
                        'loss': epoch_loss
                    })

                # Send update at specified interval
                if (epoch + 1) % self._config.update_interval == 0:
                    update = TrainingUpdate(
                        epoch=epoch + 1,
                        loss=epoch_loss,
                        gradient_norms=self._model.get_gradient_norms(),
                        learning_rate=self._config.learning_rate,
                        optimizer=self._config.optimizer
                    )
                    self._update_queue.put(update)

                    # Also call callback if provided
                    if self._update_callback:
                        self._update_callback(update)

            # Training complete
            if not self._stop_event.is_set():
                update = TrainingUpdate(
                    epoch=self._config.epochs,
                    loss=self._state.current_loss,
                    gradient_norms=self._model.get_gradient_norms(),
                    learning_rate=self._config.learning_rate,
                    optimizer=self._config.optimizer,
                    is_complete=True
                )
                self._update_queue.put(update)
                if self._update_callback:
                    self._update_callback(update)

        except Exception as e:
            # Send error update
            update = TrainingUpdate(
                epoch=self._state.epoch,
                loss=self._state.current_loss,
                gradient_norms={},
                learning_rate=self._config.learning_rate,
                optimizer=self._config.optimizer,
                error=str(e)
            )
            self._update_queue.put(update)
            if self._update_callback:
                self._update_callback(update)

        finally:
            with self._lock:
                self._state.is_training = False


# Global training runner instance (singleton for simple use case)
_global_runner: Optional[TrainingRunner] = None
_runner_lock = threading.Lock()


def get_training_runner() -> TrainingRunner:
    """Get or create the global training runner."""
    global _global_runner
    with _runner_lock:
        if _global_runner is None:
            _global_runner = TrainingRunner()
        return _global_runner


def reset_training_runner():
    """Reset the global training runner."""
    global _global_runner
    with _runner_lock:
        if _global_runner:
            _global_runner.reset()
        _global_runner = TrainingRunner()
