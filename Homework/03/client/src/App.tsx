import React, { useState, useCallback } from 'react';
import { GridMap } from './components/GridMap';
import { TrendMonitor } from './components/TrendMonitor';
import { MetricsPanel } from './components/MetricsPanel';
import { ControlPanel } from './components/ControlPanel';
import {
  useSSE,
  useTrainingControl
} from './hooks/useSSE';

function App() {
  // Training history for charts
  const [history, setHistory] = useState<Array<{ epoch: number; loss: number }>>([]);
  const [streamConnected, setStreamConnected] = useState(false);

  // SSE hook for real-time updates
  const {
    connected,
    lastUpdate,
    connect: connectSSE,
    disconnect: disconnectSSE
  } = useSSE({
    url: '/train/stream',
    autoConnect: false,
    onMessage: (data) => {
      console.log('[App] Received update:', data);
      // Update history when receiving training updates
      if (!data.is_complete && !data.error) {
        setHistory(prev => {
          const newHistory = [...prev, { epoch: data.epoch, loss: data.loss }];
          return newHistory;
        });
      }
      // Disconnect when training completes
      if (data.is_complete || data.error) {
        setTimeout(() => {
          disconnectSSE();
          setStreamConnected(false);
        }, 1000);
      }
    }
  });

  // Sync stream connection state
  React.useEffect(() => {
    setStreamConnected(connected);
  }, [connected]);

  // Training control hook
  const {
    status,
    startTraining,
    stopTraining,
    resetTraining
  } = useTrainingControl();

  // Handle training start
  const handleStartTraining = useCallback(async (params: {
    learning_rate: number;
    optimizer: 'sgd' | 'adam' | 'rmsprop';
    epochs: number;
    update_interval: number;
  }) => {
    console.log('[App] Starting training with params:', params);
    // Clear history
    setHistory([]);

    // Start training
    const result = await startTraining(params);

    if (result.success) {
      console.log('[App] Training started successfully, connecting to SSE stream...');
      // Connect to SSE stream after training starts
      setTimeout(() => {
        connectSSE();
      }, 300);
    } else {
      console.error('[App] Failed to start training:', result.error);
    }
  }, [startTraining, connectSSE]);

  // Handle reset
  const handleReset = useCallback(async () => {
    await resetTraining();
    setHistory([]);
  }, [resetTraining]);

  // Extract gradient norms from last update
  const gradientNorms = lastUpdate?.gradient_norms || {};

  return (
    <div className="dashboard-grid">
      {/* Header Bar */}
      <header className="header-bar">
        <div className="header-title">
          <div className="header-logo">GP</div>
          <div>
            <div className="header-name">GridPulse Optimizer</div>
            <div className="header-subtitle">Power Distribution Control System</div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div className={`status-indicator ${streamConnected ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            {streamConnected ? 'STREAM ACTIVE' : 'STREAM OFFLINE'}
          </div>
          <div className={`status-indicator ${status?.is_training ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            {status?.is_training ? 'MODEL TRAINING' : 'MODEL READY'}
          </div>
        </div>
      </header>

      {/* Main Visualization Area */}
      <main className="main-viz">
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Grid Topology</span>
          </div>
          <GridMap
            activeNode={status?.is_training ? 'hidden-0' : undefined}
            showConnections={true}
            height={450}
          />
        </div>

        <TrendMonitor
          data={history}
          title="System Imbalance (Residual Error)"
          height={220}
          showTarget={true}
        />
      </main>

      {/* Side Panel - Controls & Metrics */}
      <aside className="side-panel">
        <ControlPanel
          isTraining={status?.is_training || false}
          onStartTraining={handleStartTraining}
          onStopTraining={stopTraining}
          onReset={handleReset}
        />

        {status && (
          <MetricsPanel
            epoch={status.epoch}
            totalEpochs={status.total_epochs}
            currentLoss={status.current_loss}
            learningRate={status.learning_rate}
            optimizer={status.optimizer}
            gradientNorms={gradientNorms}
          />
        )}
      </aside>

      {/* Footer Bar */}
      <footer className="footer-bar">
        <div className="footer-status">
          <div className="footer-item">
            <span>API: </span>
            <span className={streamConnected ? 'text-success' : 'text-muted'}>
              {streamConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>
          <div className="footer-item">
            <span>Epoch: </span>
            <span className="mono">{status?.epoch || 0}</span>
          </div>
          <div className="footer-item">
            <span>Loss: </span>
            <span className="mono">{status?.current_loss.toFixed(6) || '0.000000'}</span>
          </div>
        </div>
        <div className="footer-item">
          <span>GridPulse Optimizer v1.0.0 | Neural Network Training Interface</span>
        </div>
      </footer>
    </div>
  );
}

export default App;