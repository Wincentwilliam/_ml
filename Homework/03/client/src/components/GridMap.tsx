import { useMemo } from 'react';

interface GridNode {
  id: string;
  type: 'input' | 'hidden' | 'output';
  label: string;
  x: number;  // Percentage position
  y: number;  // Percentage position
  value?: number;
  gradient?: number;
}

interface GridMapProps {
  nodes?: GridNode[];
  activeNode?: string;
  showConnections?: boolean;
  height?: number;
}

/**
 * Default network topology for power grid visualization.
 * Maps neural network layers to grid control stations.
 */
const DEFAULT_NODES: GridNode[] = [
  // Input layer - Sensor stations
  { id: 'input-0', type: 'input', label: 'Energy Load', x: 10, y: 20 },
  { id: 'input-1', type: 'input', label: 'Temperature', x: 10, y: 50 },
  { id: 'input-2', type: 'input', label: 'Cost', x: 10, y: 80 },

  // Hidden layer 1 - Control stations
  { id: 'hidden-0', type: 'hidden', label: 'Station A', x: 35, y: 15 },
  { id: 'hidden-1', type: 'hidden', label: 'Station B', x: 35, y: 38 },
  { id: 'hidden-2', type: 'hidden', label: 'Station C', x: 35, y: 62 },
  { id: 'hidden-3', type: 'hidden', label: 'Station D', x: 35, y: 85 },

  // Hidden layer 2 - Distribution hubs
  { id: 'hidden-4', type: 'hidden', label: 'Hub Alpha', x: 60, y: 30 },
  { id: 'hidden-5', type: 'hidden', label: 'Hub Beta', x: 60, y: 70 },

  // Output layer - Dispatch center
  { id: 'output-0', type: 'output', label: 'Power Dispatch', x: 85, y: 50 }
];

/**
 * Generate connections between layers.
 */
function generateConnections(nodes: GridNode[]): Array<{ from: string; to: string }> {
  const connections: Array<{ from: string; to: string }> = [];
  const layers = ['input', 'hidden', 'output'];

  for (let i = 0; i < layers.length - 1; i++) {
    const currentLayer = nodes.filter(n => n.type === layers[i]);
    const nextLayer = nodes.filter(n => n.type === layers[i + 1] ||
      (layers[i] === 'hidden' && n.type === 'hidden' &&
       parseInt(n.id.split('-')[1]) >= 4));

    // For hidden layer, split into two groups
    if (layers[i] === 'hidden') {
      const firstHidden = nodes.filter(n => n.type === 'hidden' && parseInt(n.id.split('-')[1]) < 4);
      const secondHidden = nodes.filter(n => n.type === 'hidden' && parseInt(n.id.split('-')[1]) >= 4);

      firstHidden.forEach(from => {
        secondHidden.forEach(to => {
          connections.push({ from: from.id, to: to.id });
        });
      });

      secondHidden.forEach(from => {
        const outputs = nodes.filter(n => n.type === 'output');
        outputs.forEach(to => {
          connections.push({ from: from.id, to: to.id });
        });
      });
    } else {
      currentLayer.forEach(from => {
        nextLayer.forEach(to => {
          connections.push({ from: from.id, to: to.id });
        });
      });
    }
  }

  return connections;
}

/**
 * Visual representation of the neural network as grid control stations.
 * Clean, modular layout avoiding spaghetti connections.
 */
export function GridMap({
  nodes = DEFAULT_NODES,
  activeNode,
  showConnections = true,
  height = 400
}: GridMapProps) {
  const connections = useMemo(() => generateConnections(nodes), [nodes]);

  const getNodeColor = (node: GridNode) => {
    if (node.id === activeNode) return '#fbbf24';
    switch (node.type) {
      case 'input': return '#3b82f6';
      case 'output': return '#fbbf24';
      default: return '#64748b';
    }
  };

  const getNodeSize = (node: GridNode) => {
    if (node.type === 'output') return 70;
    if (node.type === 'input') return 55;
    return 50;
  };

  return (
    <div className="grid-map" style={{ height }}>
      {/* Connection lines */}
      {showConnections && (
        <svg className="grid-connections" width="100%" height="100%">
          {connections.map((conn, idx) => {
            const fromNode = nodes.find(n => n.id === conn.from);
            const toNode = nodes.find(n => n.id === conn.to);
            if (!fromNode || !toNode) return null;

            return (
              <line
                key={idx}
                x1={`${fromNode.x}%`}
                y1={`${fromNode.y}%`}
                x2={`${toNode.x}%`}
                y2={`${toNode.y}%`}
                className={activeNode ? 'active' : ''}
              />
            );
          })}
        </svg>
      )}

      {/* Nodes */}
      {nodes.map((node) => {
        const size = getNodeSize(node);
        const color = getNodeColor(node);

        return (
          <div
            key={node.id}
            className={`grid-node ${node.type} ${activeNode === node.id ? 'active' : ''}`}
            style={{
              left: `calc(${node.x}% - ${size / 2}px)`,
              top: `calc(${node.y}% - ${size / 2}px)`,
              width: size,
              height: size,
              borderColor: color
            }}
          >
            {node.value !== undefined && (
              <span className="grid-node-value" style={{ color }}>
                {node.value.toFixed(3)}
              </span>
            )}
            <span className="grid-node-label">{node.label}</span>
          </div>
        );
      })}

      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: '12px',
        left: '12px',
        display: 'flex',
        gap: '16px',
        fontSize: '10px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '12px', height: '12px', background: '#3b82f6', borderRadius: '2px' }} />
          <span className="text-muted">INPUT</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '12px', height: '12px', background: '#64748b', borderRadius: '2px' }} />
          <span className="text-muted">HIDDEN</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '12px', height: '12px', background: '#fbbf24', borderRadius: '2px' }} />
          <span className="text-muted">OUTPUT</span>
        </div>
      </div>
    </div>
  );
}