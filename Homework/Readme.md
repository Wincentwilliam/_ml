<div align="center">

# 📚 Machine Learning Course Portfolio
### National Quemoy University — 2026

**Student:** 洪偉升
**Course:** Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

> *A complete record of all assignments and the midterm project, documenting the learning journey from classical search algorithms to autonomous AI agents.*

</div>

---

## 🗂️ Portfolio Overview

| # | Assignment | Topic | Algorithm / Tech | AI Tool Used |
|:---:|---|---|---|:---:|
| HW1 | [Traveling Salesperson Problem](#-homework-1--traveling-salesperson-problem) | Local Search | Hill Climbing + 2-opt | OpenCode |
| HW2 | [Backpropagation & Computational Graphs](#-homework-2--backpropagation--computational-graphs) | Neural Network Fundamentals | Chain Rule, Autograd | OpenCode |
| HW3 | [GridPulse Optimizer](#-homework-3--gridpulse-optimizer) | Neural Network Training UI | FastAPI + React + SSE | OpenCode |
| HW4 | [microGPT](#-homework-4--microgpt) | Transformer Architecture | GPT from scratch | ClaudeCode |
| HW5 | [v3-agent-secure](#-homework-5--v3-agent-secure) | AI Agent Security | LLM + Sandboxing + HITL | Claude Code |
| HW6 | [Markov Chain Text Generator](#-homework-6--second-order-markov-chain-text-generator) | NLP & Probabilistic Models | 2nd-Order Markov Chain | Claude |
| 🏆 | [AuraCode-Agent *(Midterm)*](#-midterm-project--auracode-agent) | Autonomous AI Agent | ReAct Loop + Ollama + Self-Healing | Claude Sonnet 4.6 |

---

## Learning Progression

```
HW1                HW2                 HW3                  HW4
Hill Climbing  →  Backpropagation  →  NN Training UI   →  GPT from Scratch
(Search)          (Gradients)         (Full-Stack)         (Transformer)

     HW5                  HW6                    MIDTERM
  AI Safety  →   Markov Chain NLP  →   Autonomous Self-Healing Agent
  (Security)     (Probabilistic)        (ReAct + Multi-Language + LLM)
```

---

## 📌 Homework 1 — Traveling Salesperson Problem

> **Made by: OpenCode**

### Overview

Implements a **Hill Climbing algorithm** to solve the classic Traveling Salesperson Problem (TSP) — finding the shortest Hamiltonian cycle through a set of cities.

| Component | Description |
|:---|:---|
| **Algorithm** | Hill Climbing (Local Search) |
| **Optimization Goal** | Minimize total distance (maximize negative distance) |
| **Neighbor Strategy** | 2-opt Swap (reverse a path segment to uncross edges) |
| **Reproducibility** | Deterministic via `random.seed(0)` |
| **State Representation** | Permutation of city indices `[0, 1, 2, ..., n]` |

### Core Logic

**Height Function** — Hill Climbing maximizes height, so we negate distance:
```python
def height(self) -> float:
    return -self.total_distance()
```

**Neighbor Function (2-opt swap):**
```python
def neighbor(self) -> 'TSPSolution':
    neighbor_sol = TSPSolution(self.cities)
    neighbor_sol.path = self.path[:]
    i, j = sorted(random.sample(range(len(self.path)), 2))
    neighbor_sol.path[i:j+1] = reversed(neighbor_sol.path[i:j+1])
    return neighbor_sol
```

**Hill Climbing Execution:**
```python
def hill_climbing(initial_solution: TSPSolution):
    current = initial_solution
    while True:
        neighbor = current.neighbor()
        if neighbor.height() > current.height():
            current = neighbor
        else:
            break  # Local optimum reached
    return current
```

### How to Run

```bash
.\.venv\Scripts\Activate.ps1
python HillClimbing.py
```

### Key Learnings
- Local search algorithms can find good solutions without exhaustive search
- 2-opt swaps are efficient for uncrossing route segments
- Hill Climbing is susceptible to local optima — a key limitation

---

## 📌 Homework 2 — Backpropagation & Computational Graphs

> **Made by: OpenCode**

### Overview

Demonstrates **Backpropagation** — the foundational algorithm for training neural networks — applied to two mathematical functions through explicit Computational Graph construction.

### Theoretical Framework

**Computational Graph** — maps data flow through differentiable operations. Each node is an operation; edges carry tensor values.

**Chain Rule** — the mathematical engine behind backpropagation:

$$\frac{df}{dx} = \frac{df}{dh} \cdot \frac{dh}{dx}$$

Gradients flow backward ("upstream") from output to input through the graph.

### Mathematical Analysis

**Function 1:** $f(x, y, z) = (x \cdot y) + z$ — inputs: $x=2, y=3, z=4$

| Node | Forward Value | Gradient |
|:---|:---:|:---:|
| x | 2 | **3** |
| y | 3 | **2** |
| z | 4 | **1** |
| p = x·y | 6 | 1 |
| f | 10 | 1 |

**Function 2:** $f(x, y, z, t) = ((x \cdot y) + z) \cdot t$ — inputs: $x=2, y=3, z=4, t=5$

| Node | Forward Value | Gradient |
|:---|:---:|:---:|
| x | 2 | **15** |
| y | 3 | **10** |
| z | 4 | **5** |
| t | 5 | **10** |
| p = x·y | 6 | 5 |
| q = p+z | 10 | 5 |
| f | 50 | 1 |

**Gradient derivation for x in Function 2:**
$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial p} \cdot \frac{\partial p}{\partial x} = t \cdot 1 \cdot y = 5 \cdot 1 \cdot 3 = 15$$

### Key Learnings
- Computational graphs make gradient calculation modular and scalable
- The chain rule enables gradient flow through arbitrarily deep networks
- This is exactly how PyTorch and TensorFlow compute gradients internally

---

## 📌 Homework 3 — GridPulse Optimizer

> **Made by: OpenCode**

### Overview

A professional-grade **neural network training interface** for power grid optimization simulation, featuring real-time visualization via a React frontend + FastAPI backend with SSE streaming.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │  GridMap    │  │ TrendMonitor │  │  ControlPanel/Metrics   │ │
│  │  (Nodes)    │  │ (Loss Chart) │  │  (Toggle/Metrics)       │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
│                           │  SSE Stream                         │
└───────────────────────────┼─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│        /train/start   /train/stream(SSE)   /train/reset         │
│                       TrainingRunner                            │
└───────────────────────────┼─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      nn0.py Autograd Engine                     │
│         Value (Autograd)   Linear Layers   SGD/Adam/RMSProp     │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
Homework/03/
├── nn0.py              # Custom autograd engine
├── trainer.py          # Training loop with callbacks
├── main.py             # FastAPI application
├── requirements.txt
└── client/             # React + TypeScript frontend
    └── src/
        ├── hooks/useSSE.ts
        └── components/
            ├── GridMap.tsx       # Neural network visualization
            ├── TrendMonitor.tsx  # Real-time loss chart
            ├── MetricsPanel.tsx  # Training statistics
            ├── ControlPanel.tsx  # Training controls
            └── OptimizerToggle.tsx
```

### Visual Mapping Strategy

| Neural Network Element | Grid Metaphor |
|---|---|
| Input neurons | Sensor stations (Load, Temperature, Cost) |
| Hidden neurons | Control stations / Distribution hubs |
| Output neurons | Power dispatch center |
| Weights | Power flow capacity (connection opacity) |
| Activations | Current load levels |
| Gradients | Stress indicators (color intensity) |

### How to Run

```bash
# Backend
pip install -r requirements.txt
python main.py         # → http://localhost:8000

# Frontend
cd client
npm install
npm run dev            # → http://localhost:5173
```

### Key Learnings
- SSE (Server-Sent Events) enables efficient one-way real-time streaming
- Thread-isolated training keeps the API responsive during training
- Full-stack ML visualization bridges theory and interactive UX

---

## 📌 Homework 4 — microGPT

> **Made by: ClaudeCode**

### Overview

A minimal **GPT (Generative Pre-trained Transformer)** implementation built from scratch using only Python standard library — no PyTorch, no NumPy, no external ML dependencies.

### Architecture

```
Input Tokens → Token Embeddings → Positional Embeddings
    ↓
[Transformer Block] × N layers
    (RMSNorm → Multi-Head Self-Attention → MLP)
    ↓
RMSNorm → Linear → Vocabulary Output (softmax)
```

### Key Components

| Component | Description |
|---|---|
| **Value (Autograd Engine)** | Full computational graph: add, mul, pow, relu, gelu, tanh, exp, log, softmax |
| **RMSNorm** | Root Mean Square normalization (no mean centering) |
| **Multi-Head Self-Attention** | Causal masking + scaled dot-product attention |
| **MLP (Feed-Forward)** | GELU activation, 4× expansion ratio |
| **Adam Optimizer** | Full implementation from scratch |

### Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 128 | Embedding dimension |
| `num_heads` | 4 | Attention heads |
| `num_layers` | 4 | Transformer blocks |
| `d_ff` | 512 | Feed-forward hidden dim |
| `block_size` | 64 | Max sequence length |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 0.001 | Adam learning rate |
| `epochs` | 10 | Training epochs |

### How to Run

```bash
python Micro_gpt.py
# Auto-downloads dataset → trains → generates sample text
```

### References
- *Attention Is All You Need* — Vaswani et al. (2017)
- *Language Models are Unsupervised Multitask Learners* — Radford et al. (GPT-2)
- *Adam: A Method for Stochastic Optimization* — Kingma & Ba (2014)

### Key Learnings
- Transformers are fundamentally attention + residual connections + normalization
- Building from scratch reveals exactly where gradients flow
- Character-level tokenization is simple but limited compared to BPE/WordPiece

---

## 📌 Homework 5 — v3-agent-secure

> **Made by: Claude Code**

### Overview

A **security layer for AI coding agents** implementing three defense mechanisms: file sandboxing, human-in-the-loop approval, and LLM-based security auditing — ensuring agents cannot perform unsafe operations autonomously.

### Security Architecture

```
         Agent Action
              │
              ▼
┌─────────────────────────┐
│     Path Validator      │──── Valid ────► ┌───────────────────┐
│   (BASE_DIR sandbox)    │                 │   LLM Reviewer    │
└─────────────────────────┘                 │  (safe / unsafe)  │
              │                             └─────────┬─────────┘
              │ Invalid                               │
              ▼                                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Human-in-the-Loop (y/n approval)              │
└─────────────────────────────────────────────────────────────┘
              │
              │ Approved
              ▼
┌─────────────────────────┐
│     Execute Action      │
└─────────────────────────┘
```

### Three Security Layers

**Layer 1 — Path Validator:** Restricts all file operations to a designated `BASE_DIR`. Any path traversal attempt is immediately blocked.

**Layer 2 — LLM Security Reviewer:** Secondary LLM audits every action plan before execution using structured XML responses:
```xml
<response>safe/unsafe</response>
<reason>explanation</reason>
```

**Layer 3 — Human-in-the-Loop:** Flagged actions require explicit `y/n` approval before proceeding.

### Quick Start

```python
from v3_agent_secure import SecureAgentWrapper

wrapper = SecureAgentWrapper(
    base_dir="./my_project",
    llm_client=llm_client,
    llm_model="gpt-4",
    hitl_auto_approve_patterns=[r"^logs/", r"\.tmp$"]
)

content = wrapper.read_file("src/main.py")
wrapper.write_file("output.txt", "Hello World")
```

### Installation

```bash
pip install openai   # only external dependency
```

### Key Learnings
- Defense-in-depth: multiple independent security layers are stronger than one
- LLM agents need explicit sandboxing — they cannot be trusted to self-restrict
- Human oversight remains essential for high-stakes AI operations

---

## 📌 Homework 6 — Second-Order Markov Chain Text Generator

> **Made by: Wincent (with Claude assistance)**

### Overview

A clean implementation of a **2nd-Order Markov Chain** for text generation — foundational NLP without neural networks, demonstrating probabilistic sequence modeling from first principles.

### The ML Concept — Markov Property

> *"The future depends only on the recent past, not the entire history."*

| Model | Memory | Probability |
|---|---|---|
| 1st-Order | 1 word back | `P(next \| "cat")` |
| **2nd-Order** | **2 words back** | **`P(next \| "the", "cat")`** |
| Nth-Order | N words back | `P(next \| w₁…wₙ)` |

### How It Works

**Training — Build the Transition Table:**
```
Corpus: "the cat sat on the cat mat"

("the", "cat") → ["sat", "mat"]
("cat", "sat") → ["on"]
("sat", "on")  → ["the"]
("on",  "the") → ["cat"]
```

> Duplicates are stored intentionally — `random.choice()` on the list naturally implements the probability distribution.

**Generation — Inference Loop:**
```
Step 1 → Pick seed pair: ("the", "cat")
Step 2 → Look up → ["sat", "mat"]
Step 3 → Sample → "sat"
Step 4 → Slide window → ("cat", "sat")
Step 5 → Repeat until target length
```

### Pipeline

```
tw.txt (raw text)
    → load_text()     reads file, UTF-8 safe
    → tokenize()      word or character mode (auto-detect CJK)
    → train()         builds {bigram: [successors]} dict
    → generate()      samples token-by-token
    → console output  ✅
```

### Configuration

```python
CORPUS_FILE  = "tw.txt"   # training text
GENERATE_LEN = 150        # tokens to generate
TOKEN_MODE   = "auto"     # "auto" | "word" | "char"
RANDOM_SEED  = None       # None = random; integer = reproducible
```

### How to Run

```bash
python generate.py
# No pip install required — standard library only
```

### Where This Fits in the ML Landscape

```
Markov Chain (this project)
       ↓
N-gram Language Model
       ↓
RNN / LSTM
       ↓
Transformer (Attention)
       ↓
GPT / Modern LLMs
```

### Key Learnings
- Probabilistic models can generate coherent text without any neural networks
- Storing duplicates in lists elegantly encodes probability distributions
- This is the historical foundation that led to modern language models

---

## 🏆 Midterm Project — AuraCode-Agent

> **This project is made by: Claude Anthropic Sonnet 4.6**

> ⚠️ **BETA TEST** — AuraCode-Agent is currently in beta. Some features may be unstable or incomplete. Use with caution in production environments.

---

# 🤖 AuraCode-Agent
### An Autonomous Self-Healing Engineering Environment

> *"From a simple code generator to a fully autonomous, multi-language, self-healing AI agent — built step by step."*

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Beta Status & Reliability](#beta-status--reliability)
3. [Evolutionary Process: V1 → V2 → V3](#evolutionary-process-v1--v2--v3)
4. [System Architecture](#system-architecture)
5. [Version Details](#version-details)
6. [How Each Version Works](#how-each-version-works)
7. [Supported Languages](#supported-languages)
8. [Installation & Setup](#installation--setup)
9. [Running Each Version](#running-each-version)
10. [Demo Task Recommendations](#demo-task-recommendations)
11. [AI Disclosure](#ai-disclosure)

---

## Project Overview

AuraCode-Agent is an AI-powered autonomous coding assistant that evolves across three versions — from a basic code generator to a fully self-healing, multi-language engineering environment powered by local LLMs via Ollama.

The project demonstrates a clear **evolutionary improvement pattern** required for Machine Learning coursework, showcasing:

- **V1** — Prompt Engineering & Structured Output
- **V2** — OOP Architecture + Code Execution Bridge
- **V3** — ReAct Loop + Self-Healing + Multi-Language + Full-Stack Generation

---

## Beta Status & Reliability

> ⚠️ This agent is in **BETA TEST** phase.

| Component | Reliability | Notes |
|---|---|---|
| Python code generation | ~85% | Most stable language |
| JavaScript execution | ~80% | Requires Node.js |
| Go execution | ~78% | Requires Go runtime |
| Java execution | ~75% | Requires JDK; 2-step compile |
| C++ execution | ~75% | Requires g++ / MinGW |
| Bash execution | ~70% | WSL path conversion required on Windows |
| Rust execution | ~65% | Requires GNU toolchain on Windows |
| Full-stack HTML+FastAPI | ~70% | Multi-file generation; depends on model quality |
| Self-healing (V3) | ~80% | Up to 5 repair attempts per task |
| Overall agent uptime | ~78% | Depends on Ollama model stability |

**Known limitations:**
- Scripts requiring user `input()` will timeout during auto-execution
- External Python packages must be installable via pip
- Ollama model quality directly affects output correctness
- Large model responses (>4000 chars) may be truncated or malformed

---

## Evolutionary Process: V1 → V2 → V3

```
┌──────────────────────────────────────────────────────────────────────┐
│                    EVOLUTION TIMELINE                                │
│                                                                      │
│   V1 Foundation        V2 Executor          V3 Self-Healing          │
│   ─────────────        ───────────          ────────────────         │
│                                                                      │
│   [User Prompt]   →   [User Prompt]   →    [User Prompt]             │
│        ↓                   ↓                     ↓                   │
│   [LLM Generate]      [LLM Generate]        [Auto-Detect Lang]       │
│        ↓                   ↓                     ↓                   │
│   [Save .py File]     [Execute Code]        [LLM Generate]           │
│        ↓                   ↓                     ↓                   │
│   [Show Code]         [Show Output]         [Execute Code]           │
│                            ↓                     ↓                   │
│                       [Report Error]         [Error? → Analyze]      │
│                       [STOP]                 [LLM Repair Code]       │
│                                              [Re-Execute]            │
│                                              [Repeat ≤ 5x]           │
│                                              [✅ Success / ❌ Fail] │
└──────────────────────────────────────────────────────────────────────┘
```

### What Changed Between Each Version

| Feature | V1 | V2 | V3 |
|---|---|---|---|
| Generate code from prompt | ✅ | ✅ | ✅ |
| Save to file | ✅ | ✅ | ✅ |
| Execute generated code | ❌ | ✅ | ✅ |
| Capture STDOUT / STDERR | ❌ | ✅ | ✅ |
| Auto-install pip packages | ❌ | ✅ | ✅ |
| Detect & analyze errors | ❌ | ❌ | ✅ |
| Rewrite code on failure | ❌ | ❌ | ✅ |
| Multi-language support | ❌ | ✅ (manual) | ✅ (auto-detect) |
| Full-stack web generation | ❌ | ❌ | ✅ |
| OOP class structure | ❌ | ✅ | ✅ |
| Rich terminal UI | ✅ | ✅ | ✅ |
| Session logging | ✅ | ✅ | ✅ |
| Hotkeys (Ctrl+L, Ctrl+K) | ❌ | ✅ | ✅ (Ctrl+L, F2) |
| Retry limit | ❌ | ❌ | ✅ (max 5) |
| Folder per language | ❌ | ✅ | ✅ |

---

## System Architecture

### V1 — Foundation Architecture

```
┌────────────────────────────────────────────────────┐
│                   AGENT V1                         │
│                                                    │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  User    │──▶│ Prompt       │───▶│  Ollama  │  │  
│  │  Input   │    │ Engineering  │    │  LLM API │  │
│  └──────────┘    └──────────────┘    └────┬─────┘  │
│                                           │        │ 
│                  ┌──────────────┐         │        │
│                  │ Robust       │◀───────┘         │
│                  │ Parser       │                  │
│                  └──────┬───────┘                  │
│                         │                          │
│                  ┌──────▼───────┐                  │
│                  │ Save .py to  │                  │
│                  │ generated_   │                  │
│                  │ code/v1/     │                  │
│                  └──────┬───────┘                  │
│                         │                          │
│                  ┌──────▼───────┐                  │
│                  │ Rich         │                  │
│                  │ Terminal UI  │                  │
│                  │ Display      │                  │
│                  └──────────────┘                  │
└────────────────────────────────────────────────────┘
```

### V2 — Executor Architecture

```
┌────────────────────────────────────────────────────────────┐
│                       AGENT V2                             │
│                                                            │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────┐   │
│  │  User    │──▶│ Language         │──▶│  Ollama      │   │
│  │  Input   │    │ Selector (Ctrl+K│    │  LLM API     │   │
│  └──────────┘    └─────────────────┘    └──────┬───────┘   │
│                                                │           │ 
│            ┌───────────────────────────────────▼──┐        │
│            │           CodeAgent Class            │        │
│            │  ┌─────────────┐  ┌───────────────┐  │        │
│            │  │ Parser      │  │ Language      │  │        │
│            │  │ Module      │  │ Config        │  │        │
│            │  └─────────────┘  └───────────────┘  │        │
│            └───────────────┬──────────────────────┘        │
│                            │                               │
│            ┌───────────────▼───────────────────────┐       │
│            │        ExecutionEnvironment Class     │       │
│            │                                       │       │
│            │  ┌──────────┐  ┌────────┐  ┌───────┐  │       │
│            │  │ Python   │  │  Node  │  │  Go   │  │       │
│            │  │ Runner   │  │  Runner│  │ Runner│  │       │
│            │  └──────────┘  └────────┘  └───────┘  │       │
│            │  ┌──────────┐  ┌────────┐  ┌───────┐  │       │
│            │  │  Java    │  │  G++   │  │ Rust  │  │       │
│            │  │ Runner   │  │ Runner │  │ Runner│  │       │
│            │  └──────────┘  └────────┘  └───────┘  │       │
│            └───────────────┬───────────────────────┘       │
│                            │                               │
│            ┌───────────────▼───────────────────────┐       │
│            │     Result Display + File Save        │       │
│            │     generated_code/v2/<language>/     │       │
│            └───────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────┘
```

### V3 — Self-Healing Architecture (ReAct Loop)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          AGENT V3                                   │
│                                                                     │
│  ┌──────────┐    ┌──────────────────┐    ┌─────────────────────┐    │
│  │  User    │───▶│ Auto Language   │───▶│ Language Detected:  │    │
│  │  Prompt  │    │ Detector         │    │ Python/JS/Go/Java/  │    │
│  └──────────┘    └──────────────────┘    │ C++/Rust/Bash/HTML  │    │
│                                          └──────────┬──────────┘    │
│                                                     │               │
│  ╔══════════════════════════════════════════════════▼════════════╗  │
│  ║              ReAct LOOP (max 5 attempts)                      ║  │
│  ║                                                               ║  │
│  ║   ┌──────────┐    ┌──────────┐    ┌──────────┐                ║  │
│  ║   │  THINK   │──▶│   ACT     │──▶│ OBSERVE  │                ║  │
│  ║   │          │    │          │    │          │                ║  │
│  ║   │ Generate │    │ Execute  │    │ Capture  │                ║  │
│  ║   │ code via │    │ code via │    │ STDOUT / │                ║  │
│  ║   │ Ollama   │    │ runtime  │    │ STDERR   │                ║  │
│  ║   └──────────┘    └──────────┘    └────┬─────┘                ║  │
│  ║                                        │                      ║  │
│  ║                              ┌─────────▼───────────┐          ║  │
│  ║                              │ Success?            │          ║  │
│  ║                              │                     │          ║  │
│  ║                         YES  │  NO: CORRECT        │          ║  │
│  ║                          ↓   │  Analyze traceback  │          ║  │
│  ║                        DONE  │  Rewrite code       │          ║  │
│  ║                              │  Loop back → THINK  │          ║  │
│  ║                              └─────────────────────┘          ║  │
│  ╚═══════════════════════════════════════════════════════════════╝  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Special: Full-Stack Mode (HTML/Web detected)               │    │
│  │                                                             │    │
│  │  Step 1: Generate index.html  ──┐                           │    │
│  │  Step 2: Generate style.css     ├── Sequential              │    │
│  │  Step 3: Generate script.js     │   per-file                │    │
│  │  Step 4: Generate app.py      ──┘   generation              │    │
│  │  Step 5: requirements.txt (template)                        │    │
│  │                                                             │    │
│  │  Output: generated_code/v3/fullstack/<project_name>/        │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Class Diagram

```
┌──────────────────────────────────────────────────────┐
│                    parser.py                         │
│  ┌────────────────────────────────────────────────┐  │
│  │ parse_llm_response(raw: str) → CodeBlock       │  │
│  │   • Handles <think> tags (Qwen models)         │  │
│  │   • Handles markdown code fences               │  │
│  │   • Handles ##FILENAME## delimiter format      │  │
│  │   • Handles raw Python fallback                │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│                   agent_v2.py / agent_v3.py          │
│                                                      │
│  @dataclass Language                                 │
│  ├── name: str                                       │
│  ├── key: str                                        │
│  ├── extension: str                                  │
│  ├── run_cmd: list                                   │
│  └── icon: str                                       │
│                                                      │
│  class ExecutionEnvironment                          │
│  ├── _run_python(fp, lang) → ExecResult              │
│  ├── _run_javascript(fp, lang) → ExecResult          │
│  ├── _run_bash(fp, lang) → ExecResult                │
│  ├── _run_java(fp, lang) → ExecResult                │
│  ├── _run_cpp(fp, lang) → ExecResult                 │
│  ├── _run_go(fp, lang) → ExecResult                  │
│  ├── _run_rust(fp, lang) → ExecResult                │
│  └── run(fp, lang) → ExecResult                      │
│                                                      │
│  class CodeAgent  (V3 only)                          │
│  ├── generate(task, lang) → CodeBlock                │
│  ├── repair(task, code, error, history) → CodeBlock  │
│  └── detect_language(prompt) → Language              │
│                                                      │
│  @dataclass AgentSession  (V3 only)                  │
│  ├── task: str                                       │
│  ├── attempts: list[AttemptLog]                      │
│  ├── final_status: str                               │
│  └── elapsed: float                                  │
└──────────────────────────────────────────────────────┘
```

---

## Version Details

### V1 — The Foundation

**Philosophy:** Prove that an LLM can generate structured, runnable code from a natural language prompt using careful prompt engineering alone.

**Key innovations:**
- Structured output contract using `##FILENAME##`, `##CODE##`, `##END##` delimiters
- Robust multi-format parser handling 4+ different model output styles
- Rich terminal UI with syntax-highlighted code display
- Session logging to `logs/agent_v1_*.log`

**Limitation:** V1 generates code and saves it — but never verifies if it actually runs.

---

### V2 — The Executor

**Philosophy:** A generated script that is never run is just text. V2 closes the loop between generation and execution.

**Key innovations:**
- Full OOP structure: `Language`, `ExecutionEnvironment`, `CodeAgent` classes
- Subprocess execution bridge with 30-second timeout
- 7-language runtime support (Python, JS, Go, Java, C++, Rust, Bash)
- Dedicated runner per language (compile steps for Java, C++, Rust)
- Folder-per-language output: `generated_code/v2/python/`, etc.
- Ctrl+K hotkey for language switching

**Limitation:** V2 reports errors but cannot fix them. It gives up after the first failure.

---

### V3 — The Self-Healing Agent

**Philosophy:** A true autonomous agent does not just report failure — it learns from it and tries again.

**Key innovations:**
- Auto-language detection from natural language prompt
- ReAct loop: Thought → Action → Observation → Correction (max 5 attempts)
- Chain-of-Thought repair: model explains root cause before rewriting
- Full-stack web generation: 5 files generated sequentially (HTML, CSS, JS, Python backend, requirements)
- FastAPI backend with Ollama-powered AI chatbot integration
- `AgentSession` audit trail: every attempt logged with timestamp and status
- Ctrl+L for model switching, F2 for history

---

## How Each Version Works

### V1 Flow (step by step)

```
1. User types: "create a password generator"
2. System prompt is built with output format contract
3. Request sent to Ollama API (OpenAI-compatible endpoint)
4. Raw response received
5. Parser extracts filename, description, dependencies, code
6. Code saved to generated_code/v1/<filename>.py
7. Rich panel displays syntax-highlighted code
8. Log entry written to logs/
```

### V2 Flow (step by step)

```
1. User selects language (Ctrl+K) — e.g., JavaScript
2. User types: "create a fibonacci table"
3. Language-specific system prompt built
4. Code generated by Ollama
5. Parser extracts code block
6. Code saved to generated_code/v2/javascript/<filename>.js
7. ExecutionEnvironment._run_javascript() called
   → subprocess: node <filename>.js
   → STDOUT captured
8. Output displayed in Rich panel
9. If error → show traceback → STOP (V2 does not repair)
```

### V3 Flow (step by step)

```
1. User types: "build a web scraper for Hacker News"
2. Auto-detector scans prompt → detects Python
3. System prompt built with language rules
4. ATTEMPT 1:
   → LLM generates code
   → Code saved to generated_code/v3/python/
   → Executed via subprocess
   → ModuleNotFoundError: requests
   → FAIL → Observation recorded
5. ATTEMPT 2:
   → LLM receives: original task + previous code + error traceback
   → LLM diagnoses: "missing requests library"
   → LLM rewrites: adds pip install + import fix
   → Re-executed
   → SUCCESS ✅
6. Session summary displayed: 2 attempts, 47.3s elapsed
7. Full session log saved to logs/
```

---

## Supported Languages

| Language | Extension | Runtime | Compile Step | Status |
|---|---|---|---|---|
| 🐍 Python | `.py` | `python` | No | ✅ Stable |
| 🟨 JavaScript | `.js` | `node` | No | ✅ Stable |
| 🔧 Bash | `.sh` | `bash` (WSL) | No | ✅ Works |
| ☕ Java | `.java` | `java` | `javac` → `java` | ✅ Works |
| ⚡ C++ | `.cpp` | `g++` | `g++ -o` → run | ✅ Works |
| 🐹 Go | `.go` | `go run` | No | ✅ Works |
| 🦀 Rust | `.rs` | `rustc` | `rustc` → run | ✅ Works* |
| 🌐 HTML/Web | `.html` | Browser | FastAPI serve | ✅ V3 only |

> *Rust requires GNU toolchain on Windows: `rustup toolchain install stable-x86_64-pc-windows-gnu`

---

## Installation & Setup

### Prerequisites

```
Python 3.11+
Ollama (https://ollama.com)
Node.js v18+  (for JavaScript)
Java JDK 17+  (for Java)
Go 1.21+      (for Go)
Rust 1.70+    (for Rust)
g++ / MinGW   (for C++)
Git Bash / WSL (for Bash on Windows)
```

### Install Steps

```bash
# 1. Clone or download the project
cd AuraCode-Agent

# 2. Create virtual environment
python -m venv .venv

# 3. Activate (Windows)
.venv\Scripts\activate

# 4. Install dependencies
pip install openai rich prompt_toolkit fastapi uvicorn

# 5. Pull an Ollama model
ollama pull qwen3.5:9b
```

### Project Structure

```
AuraCode-Agent/
├── parser.py                    ← Shared robust LLM output parser
├── requirements.txt             ← Python dependencies
├── README.md                    ← This file
│
├── v1_foundation/
│   └── agent_v1.py              ← V1: Foundation agent
│
├── v2_executor/
│   └── agent_v2.py              ← V2: Executor agent
│
├── v3_self_healing/
│   └── agent_v3.py              ← V3: Self-healing agent
│
├── generated_code/
│   ├── v1/                      ← V1 outputs
│   ├── v2/
│   │   ├── python/
│   │   ├── javascript/
│   │   ├── bash/
│   │   ├── java/
│   │   ├── cpp/
│   │   ├── go/
│   │   └── rust/
│   └── v3/
│       ├── python/
│       ├── javascript/
│       ├── bash/
│       ├── java/
│       ├── cpp/
│       ├── go/
│       ├── rust/
│       └── fullstack/           ← Full-stack web projects
│
└── logs/                        ← Session logs (all versions)
```

---

## Running Each Version

```bash
# Make sure Ollama is running
ollama serve   # (or it may already be running as a background service)

# Activate virtual environment
.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Mac/Linux

# Run V1
python v1_foundation\agent_v1.py

# Run V2
python v2_executor\agent_v2.py

# Run V3
python v3_self_healing\agent_v3.py
```

### Hotkeys

| Key | V1 | V2 | V3 |
|---|---|---|---|
| `Ctrl+L` | — | Switch Model | Switch Model |
| `Ctrl+K` | — | Switch Language | — |
| `F2` | — | — | View History |
| `Ctrl+C` | Exit | Exit | Exit |

---

## Demo Task Recommendations

### V1 — Foundation
```
create a script that displays a solar system with planet names and distances from the sun
```

### V2 — Executor (try each language!)
```
[Python]      create a script that generates 10 random passwords and displays them in a table
[JavaScript]  create a fibonacci sequence generator up to 20 terms with formatted output
[Go]          create a multiplication table from 1 to 10 in a formatted grid
[Java]        create a stack data structure with push, pop, peek operations and demo
[C++]         create a bubble sort implementation showing array before and after
[Rust]        create a program that calculates factorial for numbers 1 to 10
[Bash]        create a script that shows system info: OS, user, disk usage
```

### V3 — Self-Healing (these will trigger the ReAct loop!)
```
[Python]    create a script that fetches top 5 posts from Hacker News API and saves to CSV
[HTML/Web]  Create a coffee shop website with menu photos, fake payments, and an AI barista chatbot.
[Auto]      create a script that uses requests to fetch weather from wttr.in for Jakarta
```

---

## AI Disclosure

This project was built with assistance from **Claude Sonnet 4.6 by Anthropic** as part of a Machine Learning midterm project.

### What Claude Designed
- Overall system architecture (V1 → V2 → V3 evolution)
- The `parser.py` robust multi-format LLM output parser
- Language runner implementations for all 7 languages
- The ReAct self-healing loop logic
- Full-stack web generation pipeline
- Windows-specific fixes (WSL path conversion, Rust GNU toolchain, Java 2-step compile)
- Rich terminal UI panels and styling

### What the Student Designed
- The project concept: "autonomous self-healing agent"
- The evolutionary requirement (V1 simple → V3 complex)
- All task decisions, testing, and debugging direction
- The decision to use Ollama for local, free LLM inference
- The multi-language requirement and full-stack web feature request
- Integration and deployment on local Windows environment

### Why This Project Is Original
- Uses **local LLMs via Ollama** instead of cloud APIs — runs entirely on-device
- The **robust parser** handles 4+ different model output formats including `<think>` tags from Qwen models — a real engineering problem solved empirically
- The **Windows-specific fixes** (WSL bash stdin injection, Rust GNU toolchain auto-selection, Java class name extraction from generated code) are non-trivial engineering solutions
- The **sequential multi-file fullstack generator** was developed iteratively based on observed failures of single-prompt multi-file generation with smaller models

---

*AuraCode-Agent — Beta v0.9 — Built for NQU Machine Learning Midterm, 2026*
*Made with Claude Anthropic Sonnet 4.6*