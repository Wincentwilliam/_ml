# microGPT

A minimal GPT (Generative Pre-trained Transformer) implementation built from scratch using only standard Python libraries.

## Overview

This project implements a complete language model inspired by the GPT architecture, demonstrating the core concepts behind modern large language models. The implementation is intentionally minimal and educational, showing every component from scratch without relying on external ML libraries.

## Features

- **Custom Autograd Engine**: Complete automatic differentiation system for backpropagation
- **Transformer Architecture**: RMSNorm, Multi-Head Self-Attention, and MLP blocks
- **Adam Optimizer**: Full implementation from scratch
- **Text Generation**: Autoregressive generation with temperature sampling
- **Zero External Dependencies**: Uses only Python standard library (`math`, `random`, `os`, `urllib`)

## Files

| File | Description |
|------|-------------|
| `Micro_gpt.py` | Main implementation - contains all model components |
| `input.txt` | Training dataset (auto-downloaded if missing) |
| `ReadMe.md` | This documentation file |

## Architecture

The model follows the standard GPT decoder-only architecture:

```
Input Tokens → Token Embeddings → Positional Embeddings
    ↓
[Transformer Block] × N
    ↓
RMSNorm → Linear → Output (Vocabulary)
```

### Key Components

1. **Value Class (Autograd Engine)**
   - Tracks computational graph for automatic differentiation
   - Supports operations: add, mul, pow, relu, gelu, tanh, exp, log, softmax

2. **RMSNorm**
   - Root Mean Square Layer Normalization
   - Normalizes activations by RMS without mean centering

3. **Multi-Head Self-Attention**
   - Multiple attention heads for capturing different patterns
   - Causal masking for autoregressive generation
   - Scaled dot-product attention

4. **MLP (Feed-Forward Network)**
   - GELU activation
   - 4× expansion ratio (standard in GPT)

## Usage

Simply run:

```bash
python Micro_gpt.py
```

The script will:
1. Download the training dataset (if not present)
2. Build vocabulary from the dataset
3. Initialize the model
4. Train for 10 epochs
5. Generate and display sample text

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Embedding dimension |
| `num_heads` | 4 | Number of attention heads |
| `num_layers` | 4 | Number of transformer layers |
| `d_ff` | 512 | Feed-forward hidden dimension |
| `block_size` | 64 | Maximum sequence length |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 0.001 | Adam learning rate |
| `epochs` | 10 | Number of training epochs |

## Educational Purpose

This implementation prioritizes readability and understanding over performance. Each component is thoroughly commented to explain:

- The mathematical foundations
- How gradients flow through the computation graph
- Why each design choice was made
- Common practices in transformer models

## References

- "Attention Is All You Need" - Vaswani et al. (2017)
- "Language Models are Unsupervised Multitask Learners" - Radford et al. (GPT-2)
- "Adam: A Method for Stochastic Optimization" - Kingma & Ba (2014)
- "RMSNorm" - Zhang & Sennrich (2019)

## Limitations

- Small model size (intentionally) for educational purposes
- Character-level tokenization (not optimal for production)
- No GPU acceleration
- Slow compared to optimized implementations

## License

MIT License
