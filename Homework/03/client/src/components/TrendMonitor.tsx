import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface TrendMonitorProps {
  data: Array<{ epoch: number; loss: number }>;
  title?: string;
  height?: number;
  targetValue?: number;
  showTarget?: boolean;
}

/**
 * Live chart showing training loss as "System Imbalance".
 * Industrial-style trend monitor mimicking SCADA displays.
 */
export function TrendMonitor({
  data,
  title = 'System Imbalance',
  height = 200,
  targetValue = 0,
  showTarget = true
}: TrendMonitorProps) {
  const currentLoss = data.length > 0 ? data[data.length - 1].loss : 0;
  const maxLoss = Math.max(...data.map(d => d.loss), 0.1);
  const minLoss = Math.min(...data.map(d => d.loss), 0);

  // Determine status color based on loss magnitude
  const getStatusColor = () => {
    if (currentLoss < 0.01) return '#22c55e';  // Green - optimal
    if (currentLoss < 0.1) return '#fbbf24';   // Yellow - acceptable
    return '#ef4444';                           // Red - critical
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">{title}</span>
        <div className="data-value mono" style={{ color: getStatusColor() }}>
          {currentLoss.toFixed(6)}
        </div>
      </div>

      <div className="chart-container" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
            <XAxis
              dataKey="epoch"
              stroke="#64748b"
              tick={{ fontFamily: 'JetBrains Mono', fontSize: 10 }}
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fontFamily: 'JetBrains Mono', fontSize: 9 }}
              domain={[Math.min(minLoss, 0), maxLoss * 1.1]}
              width={55}
              tickFormatter={(value) => value.toFixed(3)}
            />
            <Tooltip
              contentStyle={{
                background: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '2px',
                fontFamily: 'JetBrains Mono',
                fontSize: '11px'
              }}
              labelStyle={{ color: '#94a3b8' }}
              itemStyle={{ color: '#fbbf24' }}
              formatter={(value: number) => [value.toFixed(6), 'Imbalance']}
            />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#fbbf24"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            {showTarget && (
              <Line
                type="monotone"
                dataKey={() => targetValue}
                stroke="#22c55e"
                strokeWidth={1}
                strokeDasharray="4 4"
                dot={false}
                isAnimationActive={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-between' }}>
        <div>
          <div className="data-label">Min Imbalance</div>
          <div className="data-value mono text-sm">{minLoss.toFixed(6)}</div>
        </div>
        <div>
          <div className="data-label">Max Imbalance</div>
          <div className="data-value mono text-sm">{maxLoss.toFixed(6)}</div>
        </div>
        <div>
          <div className="data-label">Samples</div>
          <div className="data-value mono text-sm">{data.length}</div>
        </div>
      </div>
    </div>
  );
}