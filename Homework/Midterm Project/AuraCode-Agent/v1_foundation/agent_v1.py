"""
AuraCode-Agent · Version 1.0 · The Foundation
Ollama Local Edition

Accepts a natural-language coding task, generates Python code via
a local Ollama model, and saves it to disk.

Author: AuraCode-Agent V1
"""

import sys
import json
import logging
import datetime
from pathlib import Path

# Add project root to path so we can import shared parser
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parser import parse_llm_response, deps_to_list

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule

# ─────────────────────────────────────────────
#  CONFIG  — only change MODEL_ID
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "generated_code" / "v1"
LOG_DIR         = BASE_DIR / "logs"
LOG_FILE        = LOG_DIR  / "agent_log.txt"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_ID        = "qwen3.5:9b"          # ← change model here

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
"""
AuraCode-Agent · Version 1.0 · The Foundation
Ollama Local Edition

LLM-powered Python code generator.
Generated files are saved to: generated_code/v1/

Author: AuraCode-Agent V1
"""

import sys
import json
import logging
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parser import parse_llm_response, deps_to_list

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
OUTPUT_DIR      = BASE_DIR / "generated_code" / "v1"   # ← V1 output folder
LOG_DIR         = BASE_DIR / "logs"
LOG_FILE        = LOG_DIR  / "agent_log.txt"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_ID        = "qwen3.5:9b"                          # ← change model here

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger  = logging.getLogger("AuraCode.V1")
console = Console()
client  = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
    timeout=600,   # wait up to 10 minutes for slow local models
)

# ─────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a Python code generator. Always respond using EXACTLY this structure:

##FILENAME##
<script_name_no_extension>
##DESCRIPTION##
<one sentence describing what the script does>
##DEPENDENCIES##
<comma-separated pip packages, or the word: none>
##CODE##
<complete runnable Python script here>
##END##

Critical rules:
- Begin your response with ##FILENAME## immediately — no intro text
- Do NOT use markdown fences (no ```) inside ##CODE##
- Do NOT add any text before ##FILENAME## or after ##END##
- The script in ##CODE## must be complete and runnable as-is
- Always include if __name__ == '__main__': guard
"""


# ═══════════════════════════════════════════════════════════════════
#  GENERATE
# ═══════════════════════════════════════════════════════════════════
def generate(task: str) -> dict:
    """Send task to Ollama and return parsed result."""
    logger.info("Task: %s", task[:120])
    console.print(Panel(
        Text("🧠  Sending to Ollama — generating code…", style="bold cyan"),
        title=f"[bold yellow]AuraCode V1 · {MODEL_ID}[/bold yellow]",
        border_style="yellow",
    ))

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Write a Python script that does:\n\n{task.strip()}"},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    logger.info("Response: %d chars", len(raw))

    if not raw.strip():
        raise ValueError("Model returned empty response. Try a simpler task or check Ollama.")

    return parse_llm_response(raw)


# ═══════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════
def save(parsed: dict) -> Path:
    """Save generated code + metadata to generated_code/v1/"""
    name = parsed["filename"]
    if not name.endswith(".py"):
        name += ".py"
    fp = OUTPUT_DIR / name
    fp.write_text(parsed["code"], encoding="utf-8")

    meta = {
        "agent":        "AuraCode V1",
        "model":        MODEL_ID,
        "timestamp":    datetime.datetime.now().isoformat(),
        "description":  parsed["description"],
        "dependencies": parsed["dependencies"],
    }
    (OUTPUT_DIR / name.replace(".py", "_meta.json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Saved → %s", fp)
    return fp


# ═══════════════════════════════════════════════════════════════════
#  DISPLAY
# ═══════════════════════════════════════════════════════════════════
def display(parsed: dict, fp: Path) -> None:
    console.print()
    console.print(Rule("[bold green]✅  Generation Complete[/bold green]", style="green"))
    console.print(Panel(
        f"[bold]File:[/bold]         {fp.name}\n"
        f"[bold]Saved to:[/bold]     {fp}\n"
        f"[bold]Description:[/bold]  {parsed['description']}\n"
        f"[bold]Dependencies:[/bold] {parsed['dependencies']}",
        title="[cyan]📋 Metadata[/cyan]",
        border_style="cyan",
    ))
    console.print(Panel(
        Syntax(parsed["code"], "python", theme="monokai", line_numbers=True),
        title="[magenta]🐍 Generated Code[/magenta]",
        border_style="magenta",
    ))
    deps = deps_to_list(parsed["dependencies"])
    if deps:
        console.print(Panel(
            f"[yellow]Run before executing:[/yellow]\n\n"
            f"  [bold]pip install {' '.join(deps)}[/bold]",
            title="[yellow]📦 Dependencies[/yellow]",
            border_style="yellow",
        ))


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    console.print(Panel(
        f"[bold white]AuraCode[/bold white]  [cyan]V1.0 — The Foundation[/cyan]\n"
        f"[dim]Ollama · {MODEL_ID}  ·  Output → generated_code/v1/[/dim]",
        border_style="bright_blue",
        padding=(1, 4),
    ))

    while True:
        console.print()
        task = console.input("[bold green]►  Describe your Python script:[/bold green] ").strip()
        if not task:
            continue
        if task.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break
        try:
            parsed = generate(task)
            fp     = save(parsed)
            display(parsed, fp)
        except ValueError as exc:
            logger.error("Error: %s", exc)
            console.print(Panel(str(exc)[:600], title="[red]Error[/red]", border_style="red"))
        except Exception as exc:
            logger.exception("Unexpected: %s", exc)
            console.print(f"[bold red]Error:[/bold red] {exc}")


if __name__ == "__main__":
    main()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger  = logging.getLogger("AuraCode.V1")
console = Console()
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
    timeout=600,       
)

# ─────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a Python code generator. Always respond using EXACTLY this structure:

##FILENAME##
<script_name_no_extension>
##DESCRIPTION##
<one sentence describing what the script does>
##DEPENDENCIES##
<comma-separated pip packages, or the word: none>
##CODE##
<complete runnable Python script here>
##END##

Critical rules:
- Begin your response with ##FILENAME## immediately
- Do NOT use markdown fences (no ```) inside ##CODE##
- Do NOT add any text before ##FILENAME## or after ##END##
- The script in ##CODE## must be complete and runnable as-is
"""


# ═══════════════════════════════════════════════════════════════════
#  GENERATE
# ═══════════════════════════════════════════════════════════════════
def generate(task: str) -> dict:
    """Call Ollama and return parsed structured result."""
    logger.info("Task: %s", task[:100])
    console.print(Panel(
        Text("🧠  Sending to Ollama — generating code…", style="bold cyan"),
        title=f"[bold yellow]AuraCode V1 · {MODEL_ID}[/bold yellow]",
        border_style="yellow",
    ))

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Write a Python script that does:\n\n{task.strip()}"},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    logger.info("Response: %d chars", len(raw))
    return parse_llm_response(raw)


# ═══════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════
def save(parsed: dict) -> Path:
    """Write code and metadata JSON to the output directory."""
    name = parsed["filename"]
    if not name.endswith(".py"):
        name += ".py"
    fp = OUTPUT_DIR / name
    fp.write_text(parsed["code"], encoding="utf-8")

    meta = {
        "agent":        "AuraCode V1",
        "model":        MODEL_ID,
        "timestamp":    datetime.datetime.now().isoformat(),
        "description":  parsed["description"],
        "dependencies": parsed["dependencies"],
    }
    (OUTPUT_DIR / name.replace(".py", "_meta.json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved → %s", fp)
    return fp


# ═══════════════════════════════════════════════════════════════════
#  DISPLAY
# ═══════════════════════════════════════════════════════════════════
def display(parsed: dict, fp: Path) -> None:
    console.print()
    console.print(Rule("[bold green]✅  Generation Complete[/bold green]", style="green"))
    console.print(Panel(
        f"[bold]File:[/bold]         {fp.name}\n"
        f"[bold]Description:[/bold]  {parsed['description']}\n"
        f"[bold]Dependencies:[/bold] {parsed['dependencies']}\n"
        f"[bold]Saved:[/bold]        {fp}",
        title="[cyan]📋 Metadata[/cyan]", border_style="cyan",
    ))
    console.print(Panel(
        Syntax(parsed["code"], "python", theme="monokai", line_numbers=True),
        title="[magenta]🐍 Generated Code[/magenta]", border_style="magenta",
    ))
    deps = deps_to_list(parsed["dependencies"])
    if deps:
        console.print(Panel(
            f"[yellow]Run before executing:[/yellow]\n\n  [bold]pip install {' '.join(deps)}[/bold]",
            title="[yellow]📦 Dependencies[/yellow]", border_style="yellow",
        ))


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    console.print(Panel(
        f"[bold white]AuraCode[/bold white]  [cyan]V1.0 — The Foundation[/cyan]\n"
        f"[dim]Ollama · {MODEL_ID}[/dim]",
        border_style="bright_blue", padding=(1, 4),
    ))

    while True:
        console.print()
        task = console.input("[bold green]►  Describe your Python script:[/bold green] ").strip()
        if not task:
            continue
        if task.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break
        try:
            parsed = generate(task)
            fp     = save(parsed)
            display(parsed, fp)
        except ValueError as exc:
            logger.error("Parse error: %s", exc)
            console.print(Panel(str(exc)[:600], title="[red]Parse Error[/red]", border_style="red"))
        except Exception as exc:
            logger.exception("Error: %s", exc)
            console.print(f"[bold red]Error:[/bold red] {exc}")


if __name__ == "__main__":
    main()