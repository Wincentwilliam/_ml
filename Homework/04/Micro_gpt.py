#!/usr/bin/env python3
"""
microGPT: A Minimal Transformer from Scratch
- Dependency-free Autograd engine
- Real training on input.txt
"""

import math
import random
import os

# =============================================================================
# AUTOGRAD ENGINE
# =============================================================================

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other): return self + other
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'pow')
        def _backward(): self.grad += out.grad * n * (self.data ** (n - 1))
        out._backward = _backward
        return out
    
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward(): self.grad += out.grad / self.data
        out._backward = _backward
        return out
    
    @staticmethod
    def softmax(values):
        data_list = [v.data for v in values]
        m = max(data_list)
        exps = [math.exp(d - m) for d in data_list]
        s = sum(exps)
        return [Value(e / s) for e in exps]
    
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev: build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo): v._backward()

# =============================================================================
# MODEL
# =============================================================================

class TinyGPT:
    def __init__(self, vocab_size, n_embd=128):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.wte = [[Value(random.uniform(-0.01, 0.01)) for _ in range(n_embd)] for _ in range(vocab_size)]
        self.lm_head = [[Value(random.uniform(-0.01, 0.01)) for _ in range(n_embd)] for _ in range(vocab_size)]

    # FIXED: Added 'targets' argument here
    def forward(self, xb, targets=None):
        B, T = len(xb), len(xb[0])
        logits = [[[sum(self.wte[xb[b][t]][d] * self.lm_head[v][d] for d in range(self.n_embd)) 
                    for v in range(self.vocab_size)] for t in range(T)] for b in range(B)]
        
        if targets is not None:
            loss = Value(0)
            for b in range(B):
                for t in range(T):
                    probs = Value.softmax(logits[b][t])
                    loss = loss - probs[targets[b][t]].log()
            return logits, loss * (1.0 / (B*T))
        return logits, None

# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    if not os.path.exists("input.txt"):
        print("Error: input.txt not found.")
        return

    with open("input.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for c, i in vocab.items()}
    data = [vocab[c] for c in text]
    
    model = TinyGPT(len(chars), n_embd=128)
    print(f"Training on {len(data)} characters...")

    for step in range(2001):
        start = random.randint(0, len(data) - 6)
        xb = [data[start:start+5]]
        yb = [data[start+1:start+6]]
        
        logits, loss = model.forward(xb, yb)
        
        for row in model.wte + model.lm_head:
            for val in row: val.grad = 0.0
            
        loss.backward()
        
        lr = 0.05
        for row in model.wte + model.lm_head:
            for val in row:
                val.data -= lr * val.grad
        
        if step % 500 == 0:
            print(f"Step {step} | Loss: {loss.data:.4f}")

    print("\n--- GENERATED TEXT OUTPUT ---")
    curr = [0]
    for _ in range(200):
        logits, _ = model.forward([curr])
        probs = Value.softmax(logits[0][-1])
        probs_data = [float(p.data) for p in probs]
        next_tok = random.choices(range(len(chars)), weights=probs_data)[0]
        curr.append(next_tok)
        print(id2char[next_tok], end="", flush=True)

if __name__ == "__main__":
    main()