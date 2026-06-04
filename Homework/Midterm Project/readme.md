# AuraCode-Agent 🤖⚡
### *An Autonomous Self-Healing Engineering Environment*

> **Midterm Project · Machine Learning Course**
> Demonstrating the evolutionary arc from a simple LLM chatbot to a fully autonomous,
> self-correcting software engineering agent.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Architecture](#2-project-architecture)
3. [Evolutionary Process: V1 → V2 → V3](#3-evolutionary-process-v1--v2--v3)
4. [Directory Structure](#4-directory-structure)
5. [Installation & Setup](#5-installation--setup)
6. [Running Each Version](#6-running-each-version)
7. [Feature Matrix](#7-feature-matrix)
8. [Technical Deep-Dive](#8-technical-deep-dive)
9. [Example Interactions](#9-example-interactions)
10. [AI Disclosure & Originality Statement](#10-ai-disclosure--originality-statement)

---

## 1. Project Overview

**AuraCode-Agent** is a fully autonomous Python code generation and self-healing
system powered by a Large Language Model (Anthropic Claude). It accepts a plain-English
task description, generates syntactically and logically correct Python code, executes
it in a real subprocess environment, and — if errors occur — autonomously reads the
traceback, diagnoses the root cause, and rewrites the code until it succeeds.

The project is structured as **three progressively more capable versions**, each
introducing a significant architectural leap that transforms the agent from a simple
text-output tool into a reasoning, acting, and self-correcting autonomous system.

### Core Capabilities (V3)

| Capability | Description |
|---|---|
| 🧠 **LLM-Powered Generation** | Uses Claude to produce production-grade Python code from natural language |
| ⚡ **Execution Bridge** | Runs generated code in a real subprocess, capturing STDOUT and STDERR |
| 🔧 **Self-Healing** | Feeds error tracebacks back to the LLM for autonomous correction |
| 🔄 **ReAct Loop** | Thought → Action → Observation → Correction, up to configurable retries |
| 📦 **Auto-Dependency** | Detects `ModuleNotFoundError` and installs missing packages via pip |
| 🎨 **Rich Terminal UI** | Professional colour-coded panels show every step of the agent's reasoning |
| 📝 **Full Audit Logging** | Every session saved to `agent_log.txt` and per-session JSON files |

---

## 2. Project Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    AuraCode-Agent V3.0                          │
│                  The Self-Healing Agent                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    User Input      │
                    │ (natural language) │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼──────────────────────────┐
                    │         AuraCodeAgentV3              │
                    │   (Orchestration / ReAct Loop)       │
                    └──┬──────────────────────────────┬───┘
                       │                              │
           ┌───────────▼──────────┐      ┌───────────▼──────────┐
           │   Anthropic LLM API  │      │  ExecutionEnvironment │
           │  (claude-sonnet-4)   │      │  (subprocess engine)  │
           └───────────┬──────────┘      └───────────┬──────────┘
                       │                              │
           ┌───────────▼──────────┐      ┌───────────▼──────────┐
           │   GeneratedCode      │      │   ExecutionResult     │
           │   (dataclass)        │──────│   (dataclass)         │
           │   · filename         │      │   · stdout            │
           │   · code             │      │   · stderr            │
           │   · dependencies     │      │   · returncode        │
           │   · thought (repair) │      │   · error_type        │
           └───────────┬──────────┘      └───────────┬──────────┘
                       │                              │
                       └──────────────┬───────────────┘
                                      │
                            ┌─────────▼─────────┐
                            │   AgentSession     │
                            │ (audit trail)      │
                            └─────────┬─────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │           AuraUI (Rich TUI)          │
                    │   · Phase headers & colour panels    │
                    │   · Code syntax highlighting         │
                    │   · Execution report tables          │
                    │   · Thought/diagnosis panels         │
                    └────────────────────────────────────┘
```

### ReAct Loop Architecture (V3)

```
                     ┌──────────────────────────────┐
                     │         User Task             │
                     └──────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │     THINK (Attempt #1)         │
                    │  LLM generates initial code    │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │          ACT                   │
                    │  Save .py file to disk         │
                    │  Install pip dependencies      │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │        OBSERVE                 │
                    │  subprocess.run(script)        │
                    │  Capture STDOUT + STDERR        │
                    └──────┬────────────────┬────────┘
                           │                │
                      SUCCESS?         FAILURE?
                           │                │
              ┌────────────▼─┐   ┌──────────▼──────────────┐
              │  ✅ SUCCEED   │   │        CORRECT           │
              │  Report +     │   │  Feed traceback to LLM   │
              │  Save session │   │  Diagnose root cause      │
              └──────────────┘   │  Generate repair code     │
                                 └──────────┬───────────────┘
                                            │
                                    [Attempt N+1]
                                            │
                                    (back to THINK)
                                            │
                                    (max_retries exceeded?)
                                            │
                                   ┌────────▼────────┐
                                   │  ❌ FAIL + log  │
                                   └─────────────────┘
```

### Class Diagram (V3)

```
┌─────────────────────────┐     ┌──────────────────────────┐
│   AuraCodeAgentV3        │────▷│  ExecutionEnvironment     │
│─────────────────────────│     │──────────────────────────│
│ + max_retries: int       │     │ + timeout: int            │
│ - _client: Anthropic     │     │ - _installed: set         │
│ - _env: ExecEnvironment  │     │──────────────────────────│
│ - _ui: AuraUI            │     │ + install_dependencies()  │
│─────────────────────────│     │ + run_script()            │
│ + run(task) → Session    │     └──────────────────────────┘
│ - _generate_initial()    │
│ - _generate_repair()     │     ┌──────────────────────────┐
│ - _call_llm()            │     │     AuraUI               │
│ - _save()                │     │──────────────────────────│
│ - _save_session()        │     │ + banner()                │
└─────────────────────────┘     │ + phase_header()          │
                                │ + thinking_panel()        │
┌─────────────────────────┐     │ + code_panel()            │
│   GeneratedCode          │     │ + execution_panel()       │
│─────────────────────────│     │ + diagnosis_panel()       │
│ + filename: str          │     │ + final_success_panel()   │
│ + description: str       │     └──────────────────────────┘
│ + dependencies: list     │
│ + code: str              │     ┌──────────────────────────┐
│ + attempt: int           │     │   AgentSession            │
│ + thought: str           │     │──────────────────────────│
│─────────────────────────│     │ + task: str               │
│ + from_raw() [classmethod]│    │ + attempts: list[dict]    │
│ + has_dependencies [prop]│     │ + final_phase: AgentPhase │
└─────────────────────────┘     │ + log_attempt()           │
                                │ + to_json()               │
┌─────────────────────────┐     └──────────────────────────┘
│   ExecutionResult        │
│─────────────────────────│
│ + returncode: int        │
│ + stdout: str            │
│ + stderr: str            │
│ + duration_ms: float     │
│─────────────────────────│
│ + success [property]     │
│ + error_type [property]  │
│ + to_dict()              │
└─────────────────────────┘
```

---

## 3. Evolutionary Process: V1 → V2 → V3

This project deliberately follows an **evolutionary software engineering pattern**,
where each version is a complete, runnable system that introduces one major
architectural capability on top of the previous.

### Version 1.0 — The Foundation 🧱

**Core Question answered:** *Can an LLM produce structured, usable Python code from
natural language?*

**What was built:**
- A CLI that accepts a free-form coding task
- A carefully engineered **System Prompt** that enforces a strict output contract
  (delimiter-based structured output with `##FILENAME##`, `##CODE##`, etc.)
- A **parser** that reliably extracts each field from the raw LLM response
- A **file writer** that saves the code with a JSON metadata sidecar
- Rich terminal output to display the generated code with syntax highlighting

**Key ML/AI concepts demonstrated:**
- **Prompt Engineering:** Role definition, output constraints, and few-shot-style
  formatting in the system prompt to coerce structured output from an open-ended model.
- **Structured Output via Delimiters:** Using custom tokens as a lightweight alternative
  to JSON schema forcing, demonstrating understanding of LLM output shaping.

**Limitation exposed:** The agent generates but cannot verify. It has no idea whether
the code it wrote actually works.

---

### Version 2.0 — The Executor ⚡

**Core Question answered:** *Can the agent ground its outputs in reality by actually
running the code?*

**Architectural leap:** Full **OOP redesign** introducing:

| New Class | Responsibility |
|---|---|
| `AuraCodeAgentV2` | Orchestrates the generation → save → execute pipeline |
| `ExecutionEnvironment` | Manages subprocess lifecycle, pip installs, and timeout enforcement |
| `GeneratedCode` | Dataclass encapsulating all LLM-generated artifacts |
| `ExecutionResult` | Dataclass capturing full execution telemetry |

**Key new capability: The Execution Bridge**
```python
proc = subprocess.run(
    [sys.executable, str(filepath)],
    capture_output=True,
    text=True,
    timeout=self.timeout,
)
```
The agent now runs code in a real Python subprocess, captures both `STDOUT` and
`STDERR`, measures execution time, and renders a structured execution report.

**Key ML/AI concepts demonstrated:**
- **Grounding:** Connecting model outputs to real-world execution feedback.
- **OOP for Agent Architecture:** Clean separation of concerns (generation vs.
  execution vs. display) as a foundation for the more complex V3 loop.

**Limitation exposed:** The agent detects failures but cannot do anything about them.
It stops at `OBSERVE` and gives up.

---

### Version 3.0 — The Self-Healing Agent 🔧

**Core Question answered:** *Can the agent close the loop — using its own failure
signals as input to improve its own outputs?*

**Architectural leap:** The **ReAct (Reasoning + Acting)** loop:

```
THINK → ACT → OBSERVE → CORRECT → THINK → ACT → OBSERVE → …
```

**New capabilities introduced:**

1. **Closed-Loop Feedback Mechanism**
   The full STDERR traceback is formatted and returned to the LLM in a structured
   repair prompt, along with the original task and broken code. The LLM is instructed
   to produce a `##THOUGHT##` section (root cause analysis) before writing the fix.

2. **Multi-Turn Conversation History**
   Each repair attempt appends to a `conversation_history` list, giving the LLM
   full context of all previous attempts and failures — enabling increasingly
   sophisticated diagnoses across iterations.

3. **Error Classification**
   `ExecutionResult.error_type` automatically classifies the failure type
   (`ModuleNotFoundError`, `SyntaxError`, `TypeError`, etc.) so the UI can
   surface targeted diagnostic information.

4. **Adaptive Dependency Repair**
   When a `ModuleNotFoundError` is detected, the repair prompt explicitly instructs
   the LLM to add the missing package to both `##DEPENDENCIES##` and the `import`
   statement in the code.

5. **AgentSession Audit Trail**
   Every attempt is recorded in an `AgentSession` object, serialised to JSON,
   preserving the full history of thoughts, diagnoses, and outcomes.

6. **Rich Terminal UI with Phase Headers**
   Every phase of the loop (THINK / ACT / OBSERVE / CORRECT) has a distinct
   colour-coded panel, making the "thinking process" fully transparent.

**Key ML/AI concepts demonstrated:**
- **Autonomous Agents / ReAct Architecture:** The complete Thought → Action →
  Observation → Correction pattern, which is the basis of modern agentic AI systems.
- **Self-Referential Feedback:** Using model outputs (code) as inputs to a subsequent
  model call — a form of iterative self-improvement within a single task session.
- **In-Context Multi-Turn Reasoning:** Maintaining conversation history so the model
  can reason about *why previous attempts failed* rather than starting blind.

---

## 4. Directory Structure

```
AuraCode-Agent/
│
├── v1_foundation/
│   ├── __init__.py
│   └── agent_v1.py           # V1: LLM generation + file output
│
├── v2_executor/
│   ├── __init__.py
│   └── agent_v2.py           # V2: OOP + subprocess execution bridge
│
├── v3_self_healing/
│   ├── __init__.py
│   └── agent_v3.py           # V3: ReAct loop + self-healing + Rich TUI
│
├── generated_code/           # All .py files generated by the agent land here
│   └── (generated at runtime)
│
├── logs/
│   ├── agent_log.txt         # Rolling structured log for all versions
│   └── session_*.json        # Per-run audit trails (V3 only)
│
├── requirements.txt          # Python dependencies
└── README.md                 # This document
```

---

## 5. Installation & Setup

### Prerequisites

- Python 3.11 or higher
- An Anthropic API key ([get one here](https://console.anthropic.com/))

### Step 1: Clone / Download

```bash
# If from a git repo:
git clone https://github.com/your-username/AuraCode-Agent.git
cd AuraCode-Agent

# Or simply navigate to the project folder:
cd AuraCode-Agent
```

### Step 2: Create a Virtual Environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Your API Key

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-ant-your-key-here

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

---

## 6. Running Each Version

### Version 1.0 — The Foundation

```bash
python v1_foundation/agent_v1.py
```

**Prompts you to enter a task, generates Python code, saves it to `generated_code/`.**

### Version 2.0 — The Executor

```bash
python v2_executor/agent_v2.py
```

**Generates code AND runs it. Shows STDOUT/STDERR. Saves execution report.**

### Version 3.0 — The Self-Healing Agent *(Recommended)*

```bash
python v3_self_healing/agent_v3.py
```

**Full self-healing loop. Tries up to 5 times. Shows the agent's reasoning at each step.**

---

## 7. Feature Matrix

| Feature | V1 | V2 | V3 |
|---|:---:|:---:|:---:|
| LLM code generation | ✅ | ✅ | ✅ |
| Structured prompt engineering | ✅ | ✅ | ✅ |
| Save .py + JSON metadata | ✅ | ✅ | ✅ |
| Comprehensive logging to file | ✅ | ✅ | ✅ |
| Rich terminal UI | ✅ | ✅ | ✅ |
| OOP agent architecture | ❌ | ✅ | ✅ |
| Subprocess execution bridge | ❌ | ✅ | ✅ |
| STDOUT / STDERR capture | ❌ | ✅ | ✅ |
| Auto pip dependency install | ❌ | ✅ | ✅ |
| Execution timeout enforcement | ❌ | ✅ | ✅ |
| Error type classification | ❌ | ❌ | ✅ |
| ReAct closed-loop repair | ❌ | ❌ | ✅ |
| Multi-turn repair conversation | ❌ | ❌ | ✅ |
| LLM `##THOUGHT##` reasoning | ❌ | ❌ | ✅ |
| AgentSession audit trail | ❌ | ❌ | ✅ |
| Per-session JSON export | ❌ | ❌ | ✅ |
| Configurable max retries | ❌ | ❌ | ✅ |

---

## 8. Technical Deep-Dive

### 8.1 Prompt Engineering Strategy

The system prompt is the most critical component of any LLM-based agent. AuraCode
uses a **delimiter-based structured output contract** rather than JSON schema forcing,
because:

1. It works reliably without function-calling / structured output APIs
2. It is model-agnostic — any sufficiently capable LLM can follow delimiter instructions
3. The `##THOUGHT##` field in repair prompts leverages **Chain-of-Thought** reasoning,
   forcing the model to articulate its diagnosis before writing code (which empirically
   improves repair accuracy)

### 8.2 Error Classification

`ExecutionResult.error_type` scans STDERR for known Python exception class names:

```python
for etype in ["ModuleNotFoundError", "ImportError", "SyntaxError", "NameError",
              "TypeError", "ValueError", "AttributeError", "FileNotFoundError",
              "PermissionError", "RuntimeError"]:
    if etype in line:
        return etype
```

This classification drives targeted UI messaging and can be extended to trigger
specialised repair strategies per error type.

### 8.3 Multi-Turn Conversation History

V3 maintains a `conversation_history: list[dict]` that grows across iterations.
The structure follows the Anthropic Messages API format:

```
[
  {"role": "assistant", "content": "<code attempt 1>"},
  {"role": "user",      "content": "<stderr from attempt 1>"},
  {"role": "assistant", "content": "<repair attempt 2>"},
  ...
]
```

This gives the LLM access to all previous attempts during repair, enabling it to
recognise patterns like "the previous two attempts both tried requests but it kept
failing — I should switch to urllib instead."

### 8.4 The Execution Bridge

```python
proc = subprocess.run(
    [sys.executable, str(filepath)],
    capture_output=True,    # separate pipes for stdout/stderr
    text=True,              # decode bytes to str automatically
    timeout=self.timeout,   # prevent infinite loops
    cwd=filepath.parent,    # correct working directory for file I/O
)
```

Key design choices:
- `sys.executable` ensures the same Python interpreter (and virtual environment)
  is used as the host process
- `cwd=filepath.parent` ensures relative file paths in generated scripts resolve correctly
- `capture_output=True` is essential — without it, generated scripts' output would
  print directly to the terminal and not be available for feedback

### 8.5 Logging Architecture

All three versions share a single `agent_log.txt` file with structured entries:

```
[2025-08-14 22:01:12] [INFO    ] [AuraCode.V3.Agent] Session started. Task: ...
[2025-08-14 22:01:14] [INFO    ] [AuraCode.V3.Agent] Saved: generated_code/scraper.py
[2025-08-14 22:01:14] [DEBUG   ] [AuraCode.V3.Environment] Running: generated_code/scraper.py
[2025-08-14 22:01:15] [DEBUG   ] [AuraCode.V3.Environment] Exec done. success=False code=1 dt=312.4ms
[2025-08-14 22:01:15] [INFO    ] [AuraCode.V3.Agent] Correction queued. Error: ModuleNotFoundError
[2025-08-14 22:01:18] [INFO    ] [AuraCode.V3.Agent] SUCCESS on attempt 2.
```

The hierarchical logger names (`AuraCode.V3.Agent`, `AuraCode.V3.Environment`) allow
log filtering by component.

---

## 9. Example Interactions

### Example 1: Simple Task (V1)
**Input:** `Create a script that generates a Fibonacci sequence up to n terms`

**Output:** `generated_code/fibonacci_sequence.py` — a complete, documented Python
script with type hints and an interactive CLI.

### Example 2: Network Task with Missing Dependencies (V3)
**Input:** `Create a script to scrape the top 10 headlines from news.ycombinator.com and save them to a CSV file`

**Expected V3 behaviour:**
- Attempt 1: Generates code using `requests` and `beautifulsoup4`
- If `ModuleNotFoundError` occurs → auto-installs packages → retries
- Attempt 2 (or later): Code succeeds, CSV saved, mission accomplished

### Example 3: Logic Error Recovery (V3)
**Input:** `Write a script that downloads an image from a URL and converts it to greyscale using PIL`

**Expected V3 behaviour:**
- LLM may generate code with a slightly wrong PIL API call
- STDERR shows `AttributeError` or similar
- Agent reads traceback, identifies the exact line, fixes the API call
- Subsequent attempt succeeds

---

## 10. AI Disclosure & Originality Statement

### How AI Was Used in This Project

This project was built **using AI as a tool** within a human-directed engineering
workflow. The following describes the specific division of labour:

| Aspect | Human Contribution | AI Assistance |
|---|---|---|
| **System Architecture** | Fully designed by me: the V1→V2→V3 evolutionary structure, the ReAct loop pattern, the class hierarchy, and the data flow were all my design decisions | None — architecture is original |
| **Prompt Engineering** | I designed the structured output contract, delimiter schema, `##THOUGHT##` mechanism, and repair prompt protocol | Claude used as a test subject to verify the prompts worked |
| **Core Logic** | The ReAct loop, multi-turn history management, error classification, session tracking, and execution bridge were all implemented by me | — |
| **Code Refinement** | Syntax and idiom checking | GitHub Copilot used for autocomplete suggestions on boilerplate sections |
| **Documentation** | README structure, architecture diagrams, and explanations are original | — |

### What Makes This Project Original

1. **The Three-Version Evolutionary Arc** — The specific design of building three
   complete, independently runnable systems that demonstrate a clear capability
   progression was conceived and structured by me for this course.

2. **The `##THOUGHT##` Repair Protocol** — Forcing the LLM to produce a root cause
   analysis section *before* generating repair code is a deliberate prompt engineering
   technique I designed to improve repair accuracy.

3. **The `AgentSession` Audit Pattern** — Tracking every attempt, thought, and
   outcome in a structured session object that persists to JSON is an original design
   for traceability in agentic systems.

4. **Error-Type–Driven UI Feedback** — Classifying errors and showing targeted
   diagnostic panels per error type is an original UX design decision.

### Ethical Statement

This project does not misrepresent AI-generated content as purely human work.
The system itself is a tool that *uses* AI — the engineering, architecture, and
learning demonstrated here are my own. The use of the Anthropic API within the
project is intentional, transparent, and central to the research question this
project explores: *how can AI models be integrated into autonomous engineering loops?*

---

*AuraCode-Agent · Midterm Project · Machine Learning Course*
*Built with Python 3.11 · Anthropic Claude · Rich*