# GridPulse Optimizer

A professional-grade neural network training interface for power grid optimization simulation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │  GridMap    │  │ TrendMonitor │  │  ControlPanel/Metrics   │ │
│  │  (Nodes)    │  │ (Loss Chart) │  │  (Toggle/Metrics)       │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
│                           │                                       │
│                    SSE Stream                                     │
└───────────────────────────┼───────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │  /train/    │  │  /train/     │  │  /train/                │ │
│  │  start      │  │  stream (SSE)│  │  reset                  │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
│                           │                                       │
│                  TrainingRunner                                   │
│                           │                                       │
└───────────────────────────┼───────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      nn0.py Autograd Engine                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │  Value      │  │  Linear       │  │  Optimizers             │ │
│  │  (Autograd) │  │  Layers       │  │  (SGD/Adam/RMSProp)     │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
C:\Users\Wincent\Wincent VS Code\_ml\Homework\03\
├── nn0.py              # Autograd engine (isolated, no FastAPI deps)
├── trainer.py          # Training loop wrapper with callbacks
├── main.py             # FastAPI application
├── requirements.txt    # Python dependencies
├── client/             # React frontend
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css           # Industrial theme styles
│       ├── hooks/
│       │   └── useSSE.ts       # SSE connection hooks
│       └── components/
│           ├── GridMap.tsx     # Neural network visualization
│           ├── TrendMonitor.tsx # Loss chart
│           ├── MetricsPanel.tsx # Training statistics
│           ├── ControlPanel.tsx # Training controls
│           └── OptimizerToggle.tsx # Optimizer selector
```

## Quick Start

### Backend

```bash
cd "C:\Users\Wincent\Wincent VS Code\_ml\Homework\03"
pip install -r requirements.txt
python main.py
```

Server runs at `http://localhost:8000`

API Endpoints:
- `GET /` - Health check
- `POST /train/start` - Start training (body: `{learning_rate, optimizer, epochs, update_interval}`)
- `GET /train/stream` - SSE endpoint for real-time updates
- `GET /train/status` - Current training status
- `GET /train/model` - Model state (weights, biases)
- `POST /train/stop` - Stop training
- `POST /train/reset` - Reset model weights

### Frontend

```bash
cd "C:\Users\Wincent\Wincent VS Code\_ml\Homework\03\client"
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

## Design Philosophy

### Industrial Aesthetic (SCADA/Control System)
- **Color Palette**: Slate, charcoal, utility yellow, alert orange
- **Typography**: Inter (UI), JetBrains Mono (data)
- **Layout**: Grid-based, modular panels
- **Controls**: Physical-style toggle switches, sliders with numeric displays

### Visual Mapping Strategy
- **Input neurons** → Sensor stations (Energy Load, Temperature, Cost)
- **Hidden neurons** → Control stations / Distribution hubs
- **Output neurons** → Power dispatch center
- **Weights** → Power flow capacity (shown via connection opacity)
- **Activations** → Current load levels (node values)
- **Gradients** → Stress indicators (color intensity)

## Key Features

1. **Real-time Training**: SSE streaming for live loss/gradient updates
2. **Optimizer Selection**: Physical toggle for SGD/Adam/RMSProp
3. **Live Diagnostics**: Trend monitor showing system imbalance (loss)
4. **Grid Visualization**: Clean node topology without spaghetti connections
5. **Background Training**: Thread-isolated training loop keeps API responsive

## API Integration Pattern

```typescript
// Frontend uses hooks for clean separation
const { status, startTraining, stopTraining } = useTrainingControl();
const { connected, lastUpdate } = useSSE({ onMessage: handleUpdate });
```

```python
# Backend isolates nn0.py from FastAPI
from nn0 import GridPulseNet, get_optimizer
from trainer import TrainingRunner  # Wrapper with thread management
```
