"""
main.py - FastAPI backend for GridPulse Optimizer
Provides REST API with SSE streaming for real-time training updates.
"""

import json
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from trainer import TrainingRunner, TrainingConfig, get_training_runner, reset_training_runner


# Request/Response models
class TrainStartRequest(BaseModel):
    """Request to start training."""
    learning_rate: float = Field(default=0.01, ge=0.0001, le=1.0)
    optimizer: str = Field(default="adam", pattern="^(sgd|adam|rmsprop)$")
    epochs: int = Field(default=100, ge=10, le=1000)
    update_interval: int = Field(default=5, ge=1, le=50)


class TrainStartResponse(BaseModel):
    """Response after starting training."""
    status: str
    message: str
    config: dict


class TrainStatusResponse(BaseModel):
    """Current training status."""
    is_training: bool
    epoch: int
    total_epochs: int
    current_loss: float
    learning_rate: float
    optimizer: str


class ModelStateResponse(BaseModel):
    """Serialized model state."""
    architecture: dict
    layers: list


class ResetResponse(BaseModel):
    """Response after reset."""
    status: str
    message: str


# Global runner reference
runner: Optional[TrainingRunner] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global runner
    runner = get_training_runner()
    yield
    # Cleanup on shutdown
    if runner:
        runner.stop()


# Create FastAPI app
app = FastAPI(
    title="GridPulse Optimizer API",
    description="Neural network training API for power grid optimization",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "GridPulse Optimizer API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    global runner
    return {
        "status": "healthy",
        "training_active": runner.is_training() if runner else False
    }


@app.post("/train/start", response_model=TrainStartResponse)
async def start_training(request: TrainStartRequest):
    """
    Start training the neural network.

    - **learning_rate**: Step size for optimizer (0.0001 - 1.0)
    - **optimizer**: Optimization algorithm (sgd, adam, rmsprop)
    - **epochs**: Number of training iterations (10 - 1000)
    - **update_interval**: How often to send updates (1 - 50 epochs)
    """
    global runner

    if runner is None:
        raise HTTPException(status_code=500, detail="Training runner not initialized")

    if runner.is_training():
        raise HTTPException(
            status_code=409,
            detail="Training already in progress. Stop current training first."
        )

    config = TrainingConfig(
        learning_rate=request.learning_rate,
        optimizer=request.optimizer,
        epochs=request.epochs,
        update_interval=request.update_interval
    )

    runner.start(config)

    return TrainStartResponse(
        status="started",
        message=f"Training started with {request.optimizer} optimizer",
        config={
            "learning_rate": request.learning_rate,
            "optimizer": request.optimizer,
            "epochs": request.epochs,
            "update_interval": request.update_interval
        }
    )


@app.get("/train/stream")
async def stream_training_updates():
    """
    Server-Sent Events endpoint for real-time training updates.

    Streams JSON updates containing:
    - epoch: Current training epoch
    - loss: Current loss value
    - gradient_norms: L2 norm of gradients per layer
    - learning_rate: Current learning rate
    - optimizer: Optimizer name
    - is_complete: Whether training has finished
    """
    global runner

    if runner is None:
        raise HTTPException(status_code=500, detail="Training runner not initialized")

    async def generate():
        last_epoch = 0

        while True:
            # Check for updates
            update = runner.poll_update(timeout=0.1)

            if update is not None:
                data = update.to_dict()
                yield f"data: {json.dumps(data)}\n\n"
                last_epoch = update.epoch

                # End stream on completion
                if update.is_complete or update.error:
                    break

            # Send keepalive every 30 seconds
            yield f": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/train/status", response_model=TrainStatusResponse)
async def get_training_status():
    """Get current training status."""
    global runner

    if runner is None:
        raise HTTPException(status_code=500, detail="Training runner not initialized")

    state = runner.get_state_dict()
    return TrainStatusResponse(
        is_training=state['is_training'],
        epoch=state['epoch'],
        total_epochs=state['total_epochs'],
        current_loss=state['current_loss'],
        learning_rate=state['learning_rate'],
        optimizer=state['optimizer_name']
    )


@app.get("/train/model", response_model=ModelStateResponse)
async def get_model_state():
    """Get current model state (weights, biases, architecture)."""
    global runner

    if runner is None:
        raise HTTPException(status_code=500, detail="Training runner not initialized")

    model_state = runner.get_model_state()
    return ModelStateResponse(
        architecture=model_state.get('architecture', {}),
        layers=model_state.get('layers', [])
    )


@app.get("/train/gradients")
async def get_gradient_norms():
    """Get current gradient norms for each layer."""
    global runner

    if runner is None:
        raise HTTPException(status_code=500, detail="Training runner not initialized")

    norms = runner.get_gradient_norms()
    return {"gradient_norms": norms}


@app.post("/train/reset", response_model=ResetResponse)
async def reset_training():
    """
    Stop current training and reset model weights.
    Useful for starting a new training run with different parameters.
    """
    global runner

    if runner is None:
        raise HTTPException(status_code=500, detail="Training runner not initialized")

    # Stop current training if active
    if runner.is_training():
        runner.stop()
        await asyncio.sleep(0.1)  # Give thread time to stop

    reset_training_runner()

    return ResetResponse(
        status="reset",
        message="Model weights and training state have been reset"
    )


@app.post("/train/stop")
async def stop_training():
    """Stop current training without resetting."""
    global runner

    if runner is None:
        raise HTTPException(status_code=500, detail="Training runner not initialized")

    if runner.is_training():
        runner.stop()
        return {"status": "stopped", "message": "Training stopped by user"}
    else:
        return {"status": "idle", "message": "No training was active"}


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
