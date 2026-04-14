"""
nn0.py - Minimal Autograd Engine for GridPulse Optimizer
A micrograd-style autograd engine with neural network utilities.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import random


@dataclass
class Value:
    """Scalar value with gradient tracking for backpropagation."""
    data: float
    grad: float = 0.0
    _prev: set = field(default_factory=set, repr=False, compare=False)
    _op: str = field(default='', repr=False, compare=False)
    label: str = field(default='', repr=False, compare=False)
    _backward: Callable = field(default=None, repr=False, compare=False)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # Addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _prev={self, other}, _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # Subtraction
    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        out = Value(-self.data, _prev={self}, _op='neg')

        def _backward():
            self.grad += -1.0 * out.grad
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return other + (-self)

    # Multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _prev={self, other}, _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # Division
    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    # Power
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            out = Value(self.data ** other, _prev={self}, _op=f'**{other}')

            def _backward():
                self.grad += other * (self.data ** (other - 1)) * out.grad
            out._backward = _backward
            return out
        else:
            # For non-constant powers, use exp/log trick
            out = Value(self.data ** other.data, _prev={self, other}, _op='pow')

            def _backward():
                if self.data > 0:
                    self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
                    other.grad += (self.data ** other.data * math.log(self.data)) * out.grad
            out._backward = _backward
            return out

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            return Value(other ** self.data, _prev={self}, _op=f'{other}**')
        raise NotImplementedError("Only constant base power supported")

    # Activation functions
    def relu(self):
        out = Value(max(0, self.data), _prev={self}, _op='relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1 / (1 + math.exp(-self.data))
        out = Value(sig, _prev={self}, _op='sigmoid')

        def _backward():
            self.grad += sig * (1 - sig) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, _prev={self}, _op='tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # Utility
    def exp(self):
        e = math.exp(self.data)
        out = Value(e, _prev={self}, _op='exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        l = math.log(self.data) if self.data > 0 else float('-inf')
        out = Value(l, _prev={self}, _op='log')

        def _backward():
            self.grad += (1 / self.data) * out.grad if self.data > 0 else 0
        out._backward = _backward
        return out

    def abs(self):
        out = Value(abs(self.data), _prev={self}, _op='abs')

        def _backward():
            self.grad += (1 if self.data >= 0 else -1) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Compute gradients via topological sort and backpropagation."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            if node._backward:
                node._backward()

    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = 0.0


@dataclass
class Module:
    """Base class for neural network modules."""

    def parameters(self) -> List[Value]:
        """Return all parameters (weights, biases) in the module."""
        params = []
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, Value):
                params.append(attr)
            elif isinstance(attr, list):
                params.extend(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
        return params

    def zero_grad(self):
        """Zero all gradients in the module."""
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """Linear layer: y = Wx + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        # Xavier initialization
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.W = [[Value(random.gauss(0, scale), label=f'W[{i},{j}]')
                   for j in range(out_features)]
                  for i in range(in_features)]
        if bias:
            self.b = [Value(0.0, label=f'b[{i}]') for i in range(out_features)]
        else:
            self.b = None

    def __call__(self, x: List[Value]) -> List[Value]:
        """Forward pass."""
        assert len(x) == self.in_features, f"Expected {self.in_features} inputs, got {len(x)}"
        out = []
        for j in range(self.out_features):
            val = sum(self.W[i][j] * x[i] for i in range(self.in_features))
            if self.b:
                val = val + self.b[j]
            out.append(val)
        return out

    def parameters(self) -> List[Value]:
        params = []
        for row in self.W:
            params.extend(row)
        if self.b:
            params.extend(self.b)
        return params


class Sequential(Module):
    """Sequential container for layers."""

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class GridPulseNet(Module):
    """
    Neural network for power grid optimization.
    Inputs: [Energy Load, Temperature, Cost]
    Output: [Optimal Power Dispatch]
    """

    def __init__(self, hidden_sizes: List[int] = None):
        hidden_sizes = hidden_sizes or [8, 4]
        self.layers = []

        # Input layer: 3 -> hidden[0]
        self.layers.append(Linear(3, hidden_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Output layer: last_hidden -> 1
        self.layers.append(Linear(hidden_sizes[-1], 1))

    def __call__(self, x: List[Value]) -> Value:
        """Forward pass through the network."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            # Apply ReLU to all but last layer
            if i < len(self.layers) - 1:
                h = [val.relu() for val in h]
        return h[0]  # Single output

    def forward(self, x: List[Value]) -> Value:
        return self(x)

    def parameters(self) -> List[Value]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def get_state(self) -> dict:
        """Get serializable state of the network."""
        state = {
            'architecture': {
                'input_size': 3,
                'hidden_sizes': [],
                'output_size': 1
            },
            'layers': []
        }

        for idx, layer in enumerate(self.layers):
            layer_state = {
                'index': idx,
                'type': 'Linear',
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'weights': [[w.data for w in row] for row in layer.W],
                'weight_grads': [[w.grad for w in row] for row in layer.W],
            }
            if layer.b:
                layer_state['bias'] = [b.data for b in layer.b]
                layer_state['bias_grads'] = [b.grad for b in layer.b]

            state['layers'].append(layer_state)

            if idx < len(self.layers) - 1:
                state['architecture']['hidden_sizes'].append(layer.out_features)

        return state

    def get_gradient_norms(self) -> dict:
        """Compute gradient norms for each layer."""
        norms = {}
        for idx, layer in enumerate(self.layers):
            w_norm = math.sqrt(sum(w.grad ** 2 for row in layer.W for w in row))
            norms[f'layer_{idx}_weights'] = w_norm
            if layer.b:
                b_norm = math.sqrt(sum(b.grad ** 2 for b in layer.b))
                norms[f'layer_{idx}_bias'] = b_norm
        return norms


# Optimizers
class Optimizer:
    """Base optimizer class."""

    def __init__(self, params: List[Value], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, params: List[Value], lr: float, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = {id(p): 0.0 for p in params}

    def step(self):
        for p in self.params:
            if self.momentum > 0:
                self.velocities[id(p)] = self.momentum * self.velocities[id(p)] - self.lr * p.grad
                p.data += self.velocities[id(p)]
            else:
                p.data -= self.lr * p.grad


class Adam(Optimizer):
    """Adam optimizer with adaptive learning rates."""

    def __init__(self, params: List[Value], lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {id(p): 0.0 for p in params}  # First moment
        self.v = {id(p): 0.0 for p in params}  # Second moment
        self.t = 0  # Timestep

    def step(self):
        self.t += 1
        for p in self.params:
            self.m[id(p)] = self.beta1 * self.m[id(p)] + (1 - self.beta1) * p.grad
            self.v[id(p)] = self.beta2 * self.v[id(p)] + (1 - self.beta2) * (p.grad ** 2)

            # Bias correction
            m_hat = self.m[id(p)] / (1 - self.beta1 ** self.t)
            v_hat = self.v[id(p)] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


class RMSProp(Optimizer):
    """RMSProp optimizer."""

    def __init__(self, params: List[Value], lr: float = 0.01, decay: float = 0.9,
                 eps: float = 1e-8):
        super().__init__(params, lr)
        self.decay = decay
        self.eps = eps
        self.cache = {id(p): 0.0 for p in params}

    def step(self):
        for p in self.params:
            self.cache[id(p)] = self.decay * self.cache[id(p)] + (1 - self.decay) * (p.grad ** 2)
            p.data -= self.lr * p.grad / (math.sqrt(self.cache[id(p)]) + self.eps)


def get_optimizer(name: str, params: List[Value], lr: float) -> Optimizer:
    """Factory function to create optimizers."""
    name = name.lower()
    if name == 'sgd':
        return SGD(params, lr)
    elif name == 'adam':
        return Adam(params, lr)
    elif name == 'rmsprop':
        return RMSProp(params, lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Use 'sgd', 'adam', or 'rmsprop'.")


# Loss functions
def mse_loss(pred: Value, target: Value) -> Value:
    """Mean Squared Error loss."""
    return (pred - target) ** 2


def mae_loss(pred: Value, target: Value) -> Value:
    """Mean Absolute Error loss."""
    return (pred - target).abs()


# Training utilities
def create_training_data(n_samples: int = 100) -> List[Tuple[List[float], float]]:
    """
    Generate synthetic training data for power grid optimization.
    Inputs: [Energy Load (0-1), Temperature (0-1), Cost (0-1)]
    Output: Optimal Power Dispatch (simulated)
    """
    data = []
    for _ in range(n_samples):
        load = random.uniform(0.2, 1.0)
        temp = random.uniform(0.1, 0.9)
        cost = random.uniform(0.1, 1.0)

        # Simulated optimal dispatch (nonlinear relationship)
        # Higher load -> higher dispatch
        # Higher temp -> slightly lower efficiency
        # Higher cost -> conservative dispatch
        optimal = (
            0.5 * load +
            0.2 * (1 - temp * 0.3) +
            0.3 * (1 - cost * 0.5) +
            random.gauss(0, 0.05)  # Noise
        )
        optimal = max(0.1, min(1.0, optimal))  # Clamp to valid range

        data.append(([load, temp, cost], optimal))

    return data


class TrainingState:
    """Tracks the current training state for serialization."""

    def __init__(self):
        self.is_training = False
        self.epoch = 0
        self.total_epochs = 0
        self.current_loss = 0.0
        self.learning_rate = 0.0
        self.optimizer_name = "sgd"
        self.history = []  # List of {epoch, loss, gradient_norms}

    def to_dict(self) -> dict:
        return {
            'is_training': self.is_training,
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'current_loss': self.current_loss,
            'learning_rate': self.learning_rate,
            'optimizer_name': self.optimizer_name,
            'recent_history': self.history[-10:]  # Last 10 entries
        }
