import { useState } from 'react';
import { OptimizerToggle } from './OptimizerToggle';

interface ControlPanelProps {
  isTraining: boolean;
  onStartTraining: (params: TrainingParams) => void;
  onStopTraining: () => void;
  onReset: () => void;
}

interface TrainingParams {
  learning_rate: number;
  optimizer: 'sgd' | 'adam' | 'rmsprop';
  epochs: number;
  update_interval: number;
}

/**
 * Control panel for training operations.
 * Industrial-style controls with parameter inputs.
 */
export function ControlPanel({
  isTraining,
  onStartTraining,
  onStopTraining,
  onReset
}: ControlPanelProps) {
  const [optimizer, setOptimizer] = useState<'sgd' | 'adam' | 'rmsprop'>('adam');
  const [learningRate, setLearningRate] = useState(0.01);
  const [epochs, setEpochs] = useState(100);
  const [updateInterval, setUpdateInterval] = useState(5);

  const handleStart = () => {
    onStartTraining({
      learning_rate: learningRate,
      optimizer,
      epochs,
      update_interval: updateInterval
    });
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Training Controls</span>
        <div className={`status-indicator ${isTraining ? 'online' : 'offline'}`}>
          <span className="status-dot" />
          {isTraining ? 'TRAINING' : 'IDLE'}
        </div>
      </div>

      {/* Optimizer Selection */}
      <OptimizerToggle
        value={optimizer}
        onChange={setOptimizer}
        disabled={isTraining}
      />

      {/* Learning Rate */}
      <div className="control-section">
        <label className="control-label">Learning Rate</label>
        <div className="slider-container">
          <input
            type="range"
            min="0.0001"
            max="0.1"
            step="0.0001"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            className="slider"
            disabled={isTraining}
          />
          <span className="slider-value mono">{learningRate.toFixed(4)}</span>
        </div>
      </div>

      {/* Epochs */}
      <div className="control-section">
        <label className="control-label">Training Cycles (Epochs)</label>
        <div className="slider-container">
          <input
            type="range"
            min="10"
            max="500"
            step="10"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value))}
            className="slider"
            disabled={isTraining}
          />
          <span className="slider-value mono">{epochs}</span>
        </div>
      </div>

      {/* Update Interval */}
      <div className="control-section">
        <label className="control-label">Telemetry Interval (Epochs)</label>
        <div className="slider-container">
          <input
            type="range"
            min="1"
            max="50"
            step="1"
            value={updateInterval}
            onChange={(e) => setUpdateInterval(parseInt(e.target.value))}
            className="slider"
            disabled={isTraining}
          />
          <span className="slider-value mono">{updateInterval}</span>
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '8px', marginTop: '16px' }}>
        {!isTraining ? (
          <button
            className="btn btn-primary"
            onClick={handleStart}
            style={{ flex: 1 }}
          >
            Initiate Training
          </button>
        ) : (
          <button
            className="btn btn-danger"
            onClick={onStopTraining}
            style={{ flex: 1 }}
          >
            Abort Training
          </button>
        )}
        <button
          className="btn"
          onClick={onReset}
          disabled={isTraining}
          title="Reset model weights"
        >
          Reset
        </button>
      </div>

      {/* Parameter Summary */}
      <div style={{
        marginTop: '16px',
        padding: '12px',
        background: 'var(--color-slate-900)',
        borderRadius: '2px',
        fontSize: '11px'
      }}>
        <div className="data-label" style={{ marginBottom: '8px' }}>CONFIGURATION SUMMARY</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
          <div className="metric-row" style={{ border: 'none', padding: '2px 0' }}>
            <span className="metric-name">Protocol</span>
            <span className="metric-value mono">{optimizer.toUpperCase()}</span>
          </div>
          <div className="metric-row" style={{ border: 'none', padding: '2px 0' }}>
            <span className="metric-name">Learning Rate</span>
            <span className="metric-value mono">{learningRate}</span>
          </div>
          <div className="metric-row" style={{ border: 'none', padding: '2px 0' }}>
            <span className="metric-name">Epochs</span>
            <span className="metric-value mono">{epochs}</span>
          </div>
          <div className="metric-row" style={{ border: 'none', padding: '2px 0' }}>
            <span className="metric-name">Update Every</span>
            <span className="metric-value mono">{updateInterval}</span>
          </div>
        </div>
      </div>
    </div>
  );
}