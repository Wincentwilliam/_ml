# v3-agent-secure

Security layer for AI coding agents with file sandboxing, human-in-the-loop approval, and LLM-based security auditing.

## Features

### 1. File Sandbox (Path Validation)
Restricts all file operations to a designated `BASE_DIR`. Any attempt to access files outside this boundary is immediately blocked.

### 2. Human-in-the-Loop Interception
When an action is flagged (outside BASE_DIR or flagged by LLM), the system prompts you to manually approve or deny with `y/n`.

### 3. LLM Security Reviewer
A secondary LLM layer that audits action plans before execution. Expects XML format responses:
```xml
<response>safe/unsafe</response>
<reason>explanation</reason>
```

## Installation

No external dependencies required. Just copy `v3_agent_secure.py` into your project.

```bash
# Optional: install openai for LLM integration
pip install openai
```

## Quick Start

```python
from v3_agent_secure import SecureAgentWrapper

# Your LLM client function
def llm_client(model, messages, temperature):
    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

# Create secure wrapper
wrapper = SecureAgentWrapper(
    base_dir="./my_project",
    llm_client=llm_client,
    llm_model="gpt-4"
)

# Use wrapper for secured file operations
content = wrapper.read_file("src/main.py")
wrapper.write_file("output.txt", "Hello World")
```

## API Reference

### SecureAgentWrapper

Main wrapper combining all security features.

```python
SecureAgentWrapper(
    base_dir: str,              # Directory to restrict access to
    llm_client: Callable,      # Your LLM client function
    llm_model: str = "gpt-4",   # Model for security reviews
    enable_llm_review: bool = True,      # Enable LLM audit layer
    enable_hitl: bool = True,            # Enable human-in-the-loop
    hitl_auto_approve_patterns: list = [] # Regex patterns for auto-approve
)
```

**Methods:**
- `read_file(path)` - Securely read a file
- `write_file(path, content)` - Securely write a file
- `execute_command(command)` - Securely execute shell commands
- `review_action(action_plan)` - Manually trigger LLM review
- `get_execution_log()` - Get all action logs
- `get_stats()` - Get security statistics

### PathValidator

Standalone path validation component.

```python
validator = PathValidator("./my_project")
is_safe, error = validator.validate_path("/etc/passwd")
# Returns: (False, "Path '/etc/passwd' is outside the allowed directory...")
```

### HumanInTheLoop

Standalone human approval component.

```python
hitl = HumanInTheLoop(
    enabled=True,
    auto_approve_patterns=[r"^logs/", r"\.tmp$"]
)
approved = hitl.request_approval(action_plan, reason)
```

### LLMSecurityReviewer

Standalone LLM security auditor.

```python
reviewer = LLMSecurityReviewer(
    llm_client=your_llm_client,
    model="gpt-4"
)
review = reviewer.review_action(action_plan, base_dir)
# review.is_safe -> bool
# review.reason -> str
```

### ActionPlan

Represents an action to be reviewed.

```python
action = ActionPlan(
    action_type="write_file",
    target_path="src/config.py",
    content="# Configuration file"
)
```

## Security Flow

```
Agent Action
     │
     ▼
┌─────────────────┐
│  Path Validator │ ── Valid ──► ┌───────────────┐
│  (BASE_DIR)     │              │ LLM Reviewer  │
└─────────────────┘              └───────────────┘
     │                                 │
     │ Invalid                         │ Rating
     ▼                                 ▼
┌─────────────────┐              ┌───────────────┐
│  Human-in-the   │ ◄─────────── │ safe/unsafe   │
│  Loop (y/n)     │              └───────────────┘
└─────────────────┘
     │
     │ Approved
     ▼
┌─────────────────┐
│  Execute Action │
└─────────────────┘
```

## Examples

See `example_integration.py` for complete examples:
- Basic usage with secure file operations
- Using components individually
- Custom action handling

## Configuration

### Auto-Approve Patterns

Use regex patterns to auto-approve certain paths:

```python
wrapper = SecureAgentWrapper(
    base_dir="./project",
    llm_client=llm_client,
    hitl_auto_approve_patterns=[
        r"^logs/",        # Auto-approve anything in logs/
        r"\.tmp$",        # Auto-approve .tmp files
        r"\.test\.py$",   # Auto-approve test files
    ]
)
```

### Disabling Layers

```python
# Disable LLM review (only path validation + HITL)
wrapper = SecureAgentWrapper(
    base_dir="./project",
    llm_client=llm_client,
    enable_llm_review=False
)

# Disable HITL (path validation + LLM review only)
wrapper = SecureAgentWrapper(
    base_dir="./project",
    llm_client=llm_client,
    enable_hitl=False
)
```

## Running Tests

```bash
# Run the example integration script
python example_integration.py
```

The example includes mock LLM responses so it runs without actual API calls.
