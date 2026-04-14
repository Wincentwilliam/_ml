interface MetricsPanelProps {
  epoch: number;
  totalEpochs: number;
  currentLoss: number;
  learningRate: number;
  optimizer: string;
  gradientNorms?: Record<string, number>;
}

/**
 * Real-time metrics display panel.
 * Shows current training statistics in a tabular format.
 */
export function MetricsPanel({
  epoch,
  totalEpochs,
  currentLoss,
  learningRate,
  optimizer,
  gradientNorms = {}
}: MetricsPanelProps) {
  const progress = totalEpochs > 0 ? (epoch / totalEpochs) * 100 : 0;

  // Calculate gradient statistics
  const gradientValues = Object.values(gradientNorms);
  const avgGradient = gradientValues.length > 0
    ? gradientValues.reduce((a, b) => a + b, 0) / gradientValues.length
    : 0;
  const maxGradient = gradientValues.length > 0
    ? Math.max(...gradientValues)
    : 0;

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">System Diagnostics</span>
      </div>

      {/* Progress Bar */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: '4px'
        }}>
          <span className="data-label">Training Progress</span>
          <span className="data-value mono text-sm">{progress.toFixed(1)}%</span>
        </div>
        <div style={{
          height: '4px',
          background: 'var(--color-slate-700)',
          borderRadius: '2px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${progress}%`,
            height: '100%',
            background: 'var(--color-utility-yellow)',
            transition: 'width 0.3s ease'
          }} />
        </div>
      </div>

      {/* Primary Metrics */}
      <div className="metric-row">
        <span className="metric-name">Cycle</span>
        <span className="metric-value mono">
          {epoch} / {totalEpochs}
        </span>
      </div>

      <div className="metric-row">
        <span className="metric-name">System Imbalance (Loss)</span>
        <span className="metric-value mono" style={{
          color: currentLoss < 0.01 ? '#22c55e' : currentLoss < 0.1 ? '#fbbf24' : '#ef4444'
        }}>
          {currentLoss.toFixed(6)}
        </span>
      </div>

      <div className="metric-row">
        <span className="metric-name">Control Protocol</span>
        <span className="metric-value mono">{optimizer.toUpperCase()}</span>
      </div>

      <div className="metric-row">
        <span className="metric-name">Learning Rate</span>
        <span className="metric-value mono">{learningRate.toExponential(2)}</span>
      </div>

      {/* Gradient Statistics */}
      {Object.keys(gradientNorms).length > 0 && (
        <>
          <div style={{
            height: '1px',
            background: 'var(--color-slate-700)',
            margin: '12px 0'
          }} />
          <div className="data-label" style={{ marginBottom: '8px' }}>
            Gradient Analysis
          </div>

          <div className="metric-row">
            <span className="metric-name">Avg Gradient Norm</span>
            <span className="metric-value mono">{avgGradient.toFixed(6)}</span>
          </div>

          <div className="metric-row">
            <span className="metric-name">Max Gradient Norm</span>
            <span className="metric-value mono">{maxGradient.toFixed(6)}</span>
          </div>

          {/* Per-layer gradients */}
          {Object.entries(gradientNorms).map(([layer, norm]) => (
            <div key={layer} className="metric-row" style={{ paddingLeft: '12px' }}>
              <span className="metric-name" style={{ fontSize: '11px' }}>{layer}</span>
              <span className="metric-value mono" style={{ fontSize: '11px' }}>
                {norm.toFixed(6)}
              </span>
            </div>
          ))}
        </>
      )}

      {/* Efficiency Calculation */}
      <div style={{
        marginTop: '16px',
        padding: '12px',
        background: 'rgba(34, 197, 94, 0.1)',
        border: '1px solid rgba(34, 197, 94, 0.3)',
        borderRadius: '2px'
      }}>
        <div className="data-label">Efficiency Gain</div>
        <div className="data-value mono" style={{ color: '#22c55e', fontSize: '1.5rem' }}>
          {(100 * (1 - Math.min(currentLoss, 1))).toFixed(2)}%
        </div>
      </div>
    </div>
  );
}