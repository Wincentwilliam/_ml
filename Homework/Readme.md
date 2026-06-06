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

> **Made by: Claude Anthropic Sonnet 4.6**
> ⚠️ **BETA TEST v0.9** — Some features may be unstable. Use with caution.

### Overview

An **Autonomous Self-Healing Engineering Environment** that evolves across three versions — from a basic code generator to a fully autonomous, multi-language, self-healing AI agent powered by local LLMs via Ollama.

### The Evolutionary Journey

```
┌──────────────────────────────────────────────────────────────────────┐
│                       EVOLUTION TIMELINE                             │
│                                                                      │
│  V1 Foundation        V2 Executor           V3 Self-Healing          │
│  ─────────────        ───────────           ────────────────         │
│                                                                      │
│  [User Prompt]   →   [User Prompt]    →    [User Prompt]             │
│       ↓                   ↓                      ↓                   │
│  [LLM Generate]      [LLM Generate]        [Auto-Detect Lang]        │
│       ↓                   ↓                      ↓                   │
│  [Save .py File]     [Execute Code]         [LLM Generate]           │
│       ↓                   ↓                      ↓                   │
│  [Show Code]         [Show Output]          [Execute Code]           │
│                           ↓                      ↓                   │
│                      [Report Error]          [Error? → Analyze]      │
│                      [STOP ❌]               [LLM Repair Code]       │
│                                              [Re-Execute]            │
│                                              [Repeat ≤ 5x]           │
│                                              [✅ Success / ❌ Fail] │
└──────────────────────────────────────────────────────────────────────┘
```

### Feature Comparison

| Feature | V1 | V2 | V3 |
|---|:---:|:---:|:---:|
| Generate code from prompt | ✅ | ✅ | ✅ |
| Save to file | ✅ | ✅ | ✅ |
| Execute generated code | ❌ | ✅ | ✅ |
| Capture STDOUT / STDERR | ❌ | ✅ | ✅ |
| Auto-install pip packages | ❌ | ✅ | ✅ |
| Detect & analyze errors | ❌ | ❌ | ✅ |
| Rewrite code on failure | ❌ | ❌ | ✅ |
| Multi-language support | ❌ | ✅ manual | ✅ auto |
| Full-stack web generation | ❌ | ❌ | ✅ |
| OOP architecture | ❌ | ✅ | ✅ |
| Self-healing retry loop | ❌ | ❌ | ✅ (max 5x) |
| Folder per language | ❌ | ✅ | ✅ |

### V3 — ReAct Self-Healing Loop

```
  ╔══════════════════════════════════════════════╗
  ║         ReAct LOOP (max 5 attempts)          ║
  ║                                              ║
  ║  ┌─────────┐   ┌─────────┐   ┌──────────┐    ║
  ║  │  THINK  │──▶│   ACT   │──▶│ OBSERVE │    ║
  ║  │ Generate│   │ Execute │   │ Capture  │    ║
  ║  │ via LLM │   │ runtime │   │ STDOUT/  │    ║
  ║  └─────────┘   └─────────┘   │ STDERR   │    ║
  ║       ▲                      └────┬─────┘    ║
  ║       │                           │          ║
  ║  ┌────┴──────┐         ┌──────────▼──────┐   ║
  ║  │  CORRECT  │◀── NO ──│ Exit code = 0?  │  ║
  ║  │ Rewrite   │         └──────────┬──────┘   ║
  ║  │ with fix  │                    │ YES      ║
  ║  └───────────┘                    ▼          ║
  ║                              ✅ SUCCESS      ║
  ╚══════════════════════════════════════════════╝
```

### Supported Languages

| Language | Extension | Compile | Status |
|---|---|---|---|
| 🐍 Python | `.py` | No | ✅ ~85% |
| 🟨 JavaScript | `.js` | No | ✅ ~80% |
| 🔧 Bash | `.sh` | No | ✅ ~70% |
| ☕ Java | `.java` | `javac` → `java` | ✅ ~75% |
| ⚡ C++ | `.cpp` | `g++ -o` → run | ✅ ~75% |
| 🐹 Go | `.go` | No | ✅ ~78% |
| 🦀 Rust | `.rs` | `rustc` → run | ✅ ~65% |
| 🌐 HTML/Web | `.html` | FastAPI serve | ✅ V3 only |

### Project Structure

```
AuraCode-Agent/
├── parser.py                  ← Shared robust LLM output parser
├── requirements.txt
├── README.md
├── v1_foundation/agent_v1.py  ← V1: Foundation
├── v2_executor/agent_v2.py    ← V2: Executor
├── v3_self_healing/agent_v3.py← V3: Self-Healing
├── generated_code/
│   ├── v1/
│   ├── v2/{python,javascript,bash,java,cpp,go,rust}/
│   └── v3/{python,javascript,bash,java,cpp,go,rust,fullstack}/
└── logs/
```

### How to Run

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install openai rich prompt_toolkit fastapi uvicorn

# Pull Ollama model
ollama pull qwen3.5:9b

# Run any version
python v1_foundation\agent_v1.py
python v2_executor\agent_v2.py
python v3_self_healing\agent_v3.py
```

### Demo Tasks

```bash
# V1
"create a script that displays a solar system with planet distances"

# V2 — triggers real execution output
"create a fibonacci sequence generator up to 20 terms with formatted output"

# V3 — triggers self-healing loop
"create a script that fetches top 5 posts from Hacker News API and saves to CSV"

# V3 Full-Stack
"buat website coffee shop dengan foto menu, fake payment, dan AI barista chatbot"
```

### AI Disclosure

Built with **Claude Sonnet 4.6 by Anthropic** for NQU Machine Learning Midterm 2026.

| What Claude Built | What Student Designed |
|---|---|
| System architecture (V1→V2→V3) | Project concept: "self-healing agent" |
| Robust multi-format parser | Evolutionary requirement structure |
| All 7 language runners | Task decisions and testing direction |
| ReAct loop logic | Decision to use Ollama (free, local) |
| Windows-specific fixes (WSL, Rust GNU) | Multi-language + full-stack requirements |
| Full-stack web generator | Integration & deployment on Windows |

---

## 📊 Course Learning Arc

```
Week 1-2    Week 3-4         Week 5-6           Week 7-8
  HW1          HW2              HW3                HW4
  │            │                │                  │
Search      Gradients       Full-Stack          Transformer
Algorithms  & Autograd      ML Interface        from Scratch
  │            │                │                  │
  └────────────┴────────────────┴──────────────────┘
                                                    │
                                                    ▼
                              Week 9-10          Week 11-12        MIDTERM
                                HW5                HW6               🏆
                                │                  │                 │
                             AI Safety          Probabilistic     Autonomous
                             & Security         NLP (Markov)      AI Agent
```

### Skills Acquired

| Domain | Skills |
|---|---|
| **Search & Optimization** | Hill Climbing, 2-opt, local optima awareness |
| **Neural Networks** | Backpropagation, computational graphs, chain rule |
| **Full-Stack ML** | FastAPI, React, SSE streaming, real-time visualization |
| **Transformers** | Attention mechanism, autograd from scratch, Adam optimizer |
| **AI Safety** | Sandboxing, human-in-the-loop, LLM security auditing |
| **NLP & Probability** | Markov chains, n-gram models, probabilistic generation |
| **Autonomous Agents** | ReAct loop, self-healing, multi-language execution, LLM orchestration |

---

<div align="center">

*Machine Learning Course Portfolio — NQU Spring 2026*
*Compiled with Claude Anthropic Sonnet 4.6*

</div>