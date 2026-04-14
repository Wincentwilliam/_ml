interface OptimizerToggleProps {
  value: 'sgd' | 'adam' | 'rmsprop';
  onChange: (value: 'sgd' | 'adam' | 'rmsprop') => void;
  disabled?: boolean;
  labels?: {
    sgd: string;
    adam: string;
    rmsprop: string;
  };
}

/**
 * Physical-style toggle switch for optimizer selection.
 * Mimics industrial control panel selector switches.
 */
export function OptimizerToggle({
  value,
  onChange,
  disabled = false,
  labels = {
    sgd: 'Protocol A: Steady',
    adam: 'Protocol B: Adaptive',
    rmsprop: 'Protocol C: Rapid'
  }
}: OptimizerToggleProps) {
  return (
    <div className="control-section">
      <label className="control-label">Control Protocol</label>
      <div className="toggle-group" role="radiogroup">
        <div className="toggle-option">
          <input
            type="radio"
            id="optimizer-sgd"
            name="optimizer"
            value="sgd"
            checked={value === 'sgd'}
            onChange={() => onChange('sgd')}
            disabled={disabled}
          />
          <label htmlFor="optimizer-sgd" title="Stochastic Gradient Descent">
            {labels.sgd}
          </label>
        </div>
        <div className="toggle-option">
          <input
            type="radio"
            id="optimizer-adam"
            name="optimizer"
            value="adam"
            checked={value === 'adam'}
            onChange={() => onChange('adam')}
            disabled={disabled}
          />
          <label htmlFor="optimizer-adam" title="Adam Optimizer">
            {labels.adam}
          </label>
        </div>
        <div className="toggle-option">
          <input
            type="radio"
            id="optimizer-rmsprop"
            name="optimizer"
            value="rmsprop"
            checked={value === 'rmsprop'}
            onChange={() => onChange('rmsprop')}
            disabled={disabled}
          />
          <label htmlFor="optimizer-rmsprop" title="RMSProp Optimizer">
            {labels.rmsprop}
          </label>
        </div>
      </div>
    </div>
  );
}
