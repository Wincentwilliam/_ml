"""
AuraCode-Agent · Version 2.0 PRO · The Executor
Ollama Local Edition — Professional Terminal UI

Features:
  · Multi-language: Python, JavaScript, Bash, Java, C++, Go, Rust
  · Ctrl+L  — switch Ollama model on-the-fly
  · Ctrl+K  — switch programming language
  · Ctrl+H  — view full task history
  · Ctrl+C  — exit
  · Auto-run generated code in correct runtime
  · Self-healing ReAct loop: Think → Act → Observe → Correct
  · Session audit trail saved to JSON

Author: AuraCode-Agent V3 PRO
"""

from __future__ import annotations

import sys
import json
import os
import logging
import datetime
import subprocess
import time
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parser import parse_llm_response, deps_to_list

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.align import Align
from rich import box

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.history import InMemoryHistory

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
OUTPUT_DIR      = BASE_DIR / "generated_code" / "v2"
LOG_DIR         = BASE_DIR / "logs"
LOG_FILE        = LOG_DIR  / "agent_log.txt"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
EXEC_TIMEOUT    = 30
MAX_RETRIES     = 5
IS_WINDOWS      = sys.platform == "win32"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
logger  = logging.getLogger("AuraCode.V2Pro")
console = Console()


# ═══════════════════════════════════════════════════════════════════
#  LANGUAGE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Language:
    name:      str
    key:       str
    extension: str
    icon:      str
    runtime:   str   # binary to check with shutil.which()


# Language registry — execution handled separately in Env class
LANGUAGES: dict[str, Language] = {
    "python":     Language("Python",     "python",     ".py",   "🐍", "python"),
    "javascript": Language("JavaScript", "javascript", ".js",   "🟨", "node"),
    "bash":       Language("Bash",       "bash",       ".sh",   "🔧", "bash"),
    "java":       Language("Java",       "java",       ".java", "☕", "javac"),
    "cpp":        Language("C++",        "cpp",        ".cpp",  "⚡", "g++"),
    "go":         Language("Go",         "go",         ".go",   "🐹", "go"),
    "rust":       Language("Rust",       "rust",       ".rs",   "🦀", "rustc"),
}

LANGUAGE_LIST = list(LANGUAGES.keys())


def is_runtime_available(lang: Language) -> bool:
    """Check if the language runtime is installed on this machine."""
    return shutil.which(lang.runtime) is not None


# ═══════════════════════════════════════════════════════════════════
#  OLLAMA MODEL DETECTION
# ═══════════════════════════════════════════════════════════════════

def get_ollama_models() -> list[str]:
    """Fetch list of locally available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
        models = []
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════
#  AGENT STATE
# ═══════════════════════════════════════════════════════════════════

class AgentState:
    """Holds all mutable runtime state for the agent session."""

    def __init__(self):
        self.model_id:         str          = "qwen3.5:9b"
        self.language:         Language     = LANGUAGES["python"]
        self.history:          list[dict]   = []
        self.available_models: list[str]    = []
        self._client:          Optional[OpenAI] = None

    def get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama",
                timeout=600,
            )
        return self._client

    def set_model(self, model_id: str) -> None:
        self.model_id = model_id
        self._client  = None  # force reconnect

    def add_history(self, task: str, lang: str, model: str,
                    success: bool, filepath: str) -> None:
        self.history.append({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "task":      task[:80],
            "language":  lang,
            "model":     model,
            "success":   success,
            "file":      filepath,
        })


# ═══════════════════════════════════════════════════════════════════
#  ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

class Phase(Enum):
    THINKING   = auto()
    ACTING     = auto()
    OBSERVING  = auto()
    CORRECTING = auto()
    SUCCEEDED  = auto()
    FAILED     = auto()


@dataclass
class GeneratedCode:
    filename:     str
    description:  str
    dependencies: list[str]
    code:         str
    language:     Language
    attempt:      int = 1
    filepath:     Optional[Path] = None
    thought:      str = ""

    @property
    def has_deps(self) -> bool:
        # Only Python supports pip deps
        return bool(self.dependencies) and self.language.key == "python"

    @classmethod
    def from_parsed(cls, p: dict, lang: Language,
                    attempt: int = 1) -> "GeneratedCode":
        return cls(
            filename=p["filename"],
            description=p["description"],
            dependencies=deps_to_list(p["dependencies"]),
            code=p["code"],
            language=lang,
            attempt=attempt,
            thought=p.get("thought", ""),
        )


@dataclass
class ExecResult:
    filepath:    Path
    returncode:  int
    stdout:      str
    stderr:      str
    duration_ms: float
    timed_out:   bool = False
    skipped:     bool = False

    @property
    def success(self) -> bool:
        return (self.returncode == 0 and not self.timed_out) or self.skipped

    @property
    def error_type(self) -> str:
        if self.skipped:   return "RuntimeNotInstalled"
        if self.timed_out: return "TimeoutError"
        for line in self.stderr.splitlines():
            for e in ["ModuleNotFoundError", "ImportError", "SyntaxError",
                      "IndentationError", "NameError", "TypeError", "ValueError",
                      "AttributeError", "FileNotFoundError", "RuntimeError",
                      "KeyError", "IndexError", "error:", "Error:"]:
                if e in line:
                    return e
        return "UnknownError"

    def to_dict(self) -> dict:
        return {
            "filepath":    str(self.filepath),
            "returncode":  self.returncode,
            "success":     self.success,
            "error_type":  self.error_type,
            "duration_ms": round(self.duration_ms, 2),
            "stdout":      self.stdout[:3000],
            "stderr":      self.stderr[:3000],
        }


@dataclass
class Session:
    task:           str
    language:       Language
    model_id:       str
    started_at:     datetime.datetime = field(default_factory=datetime.datetime.now)
    attempts:       list[dict]        = field(default_factory=list)
    final_phase:    Phase             = Phase.THINKING
    total_attempts: int               = 0

    def log(self, n: int, code: GeneratedCode, r: ExecResult) -> None:
        self.attempts.append({
            "attempt":     n,
            "filename":    code.filename,
            "thought":     code.thought,
            "success":     r.success,
            "error_type":  r.error_type,
            "duration_ms": r.duration_ms,
        })
        self.total_attempts = n

    def to_json(self) -> str:
        return json.dumps({
            "task":           self.task,
            "language":       self.language.name,
            "model":          self.model_id,
            "started_at":     self.started_at.isoformat(),
            "final_phase":    self.final_phase.name,
            "total_attempts": self.total_attempts,
            "attempts":       self.attempts,
        }, indent=2)


# ═══════════════════════════════════════════════════════════════════
#  EXECUTION ENVIRONMENT
#  Each language has its own proper execution strategy
# ═══════════════════════════════════════════════════════════════════

class Env:
    """
    Handles code execution for all supported languages.
    Uses direct subprocess calls — no bash wrapper to avoid PATH issues.
    """

    def __init__(self, timeout: int = EXEC_TIMEOUT):
        self.timeout   = timeout
        self.python    = sys.executable
        self._done:    set[str] = set()
        self._log      = logging.getLogger("AuraCode.V2Pro.Env")

    # ── Python deps ───────────────────────────────────────────────

    def install_python_deps(self, pkgs: list[str]) -> bool:
        new = [p for p in pkgs if p.lower() not in self._done]
        if not new:
            return True
        console.print(Panel(
            f"[yellow]📦 Installing:[/yellow] [bold]{', '.join(new)}[/bold]",
            border_style="yellow"))
        r = subprocess.run(
            [self.python, "-m", "pip", "install", "--quiet"] + new,
            capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            self._done.update(p.lower() for p in new)
            console.print("  [green]✓ Installed.[/green]")
        else:
            console.print(f"  [red]✗ Failed: {r.stderr[:100]}[/red]")
        return r.returncode == 0

    # ── Main run dispatcher ───────────────────────────────────────

    def run(self, fp: Path, lang: Language) -> ExecResult:
        """Dispatch to the correct execution strategy for each language."""

        if not is_runtime_available(lang):
            console.print(Panel(
                f"[yellow]⚠  {lang.name} runtime ({lang.runtime}) not found.\n"
                f"Code saved to: [bold]{fp}[/bold][/yellow]",
                border_style="yellow"))
            return ExecResult(fp, 0, "", "", 0.0, skipped=True)

        self._log.info("Running %s: %s", lang.name, fp)

        dispatch = {
            "python":     self._run_python,
            "javascript": self._run_javascript,
            "bash":       self._run_bash,
            "java":       self._run_java,
            "cpp":        self._run_cpp,
            "go":         self._run_go,
            "rust":       self._run_rust,
        }

        runner = dispatch.get(lang.key, self._run_generic)
        return runner(fp)

    # ── Per-language runners ──────────────────────────────────────

    def _exec(self, cmd: list[str], cwd: Path) -> tuple[int, str, str, float, bool]:
        """
        Execute a command and return (returncode, stdout, stderr, ms, timed_out).
        """
        start     = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, cwd=cwd,
                env={**os.environ},   # pass full Windows PATH
            )
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            timed_out = True
            rc, out, err = -1, "", f"[Timeout] Exceeded {self.timeout}s"
        except FileNotFoundError as exc:
            rc, out, err = -1, "", f"[RuntimeNotFound] {exc}"
        except Exception as exc:
            rc, out, err = -1, "", str(exc)
        ms = (time.perf_counter() - start) * 1000
        return rc, out, err, ms, timed_out

    def _run_python(self, fp: Path) -> ExecResult:
        rc, out, err, ms, to = self._exec([self.python, str(fp)], fp.parent)
        return ExecResult(fp, rc, out, err, ms, to)

    def _run_javascript(self, fp: Path) -> ExecResult:
        rc, out, err, ms, to = self._exec(["node", str(fp)], fp.parent)
        return ExecResult(fp, rc, out, err, ms, to)

    def _run_bash(self, fp: Path) -> ExecResult:
        import tempfile, shutil as _sh, subprocess as _sp

        if IS_WINDOWS:
            # Detect which bash: WSL (System32) or Git bash
            bash_path = shutil.which("bash") or "bash"
            is_wsl = "system32" in bash_path.lower() or "windowsapps" in bash_path.lower()

            if is_wsl:
                # WSL bash: copy script to WSL home to avoid Windows path issues
                # Write script content, pass via stdin to avoid path problems entirely
                script_content = fp.read_text(encoding="utf-8")
                start = time.perf_counter()
                timed_out = False
                try:
                    proc = subprocess.run(
                        ["bash"],
                        input=script_content,
                        capture_output=True, text=True,
                        timeout=self.timeout,
                        env={**os.environ},
                    )
                    rc, out, err = proc.returncode, proc.stdout, proc.stderr
                except subprocess.TimeoutExpired:
                    timed_out = True
                    rc, out, err = -1, "", f"[Timeout] Exceeded {self.timeout}s"
                except Exception as exc:
                    rc, out, err = -1, "", str(exc)
                ms = (time.perf_counter() - start) * 1000
                return ExecResult(fp, rc, out, err, ms, timed_out)
            else:
                # Git bash: convert C:/foo -> /c/foo
                tmp_dir = Path(tempfile.mkdtemp())
                tmp_fp  = tmp_dir / "script.sh"
                _sh.copy2(str(fp), str(tmp_fp))
                posix = tmp_fp.as_posix()
                if len(posix) > 2 and posix[1] == ":":
                    posix = "/" + posix[0].lower() + posix[2:]
                rc, out, err, ms, to = self._exec(["bash", posix], tmp_dir)
                try:
                    _sh.rmtree(str(tmp_dir))
                except Exception:
                    pass
                return ExecResult(fp, rc, out, err, ms, to)
        else:
            # Linux/Mac: run directly
            rc, out, err, ms, to = self._exec(["bash", str(fp)], fp.parent)
            return ExecResult(fp, rc, out, err, ms, to)

    def _run_go(self, fp: Path) -> ExecResult:
        rc, out, err, ms, to = self._exec(["go", "run", str(fp)], fp.parent)
        return ExecResult(fp, rc, out, err, ms, to)

    def _run_java(self, fp: Path) -> ExecResult:
        """Compile with javac then run with java."""
        # Step 1: compile
        rc, out, err, ms, to = self._exec(
            ["javac", str(fp)], fp.parent)
        if rc != 0 or to:
            return ExecResult(fp, rc, out, err, ms, to)

        # Step 2: find class name (filename without extension)
        class_name = fp.stem
        rc2, out2, err2, ms2, to2 = self._exec(
            ["java", "-cp", str(fp.parent), class_name], fp.parent)
        return ExecResult(fp, rc2, out2, err2, ms + ms2, to2)

    def _run_cpp(self, fp: Path) -> ExecResult:
        """Compile with g++ then run the binary."""
        # Output binary path (add .exe on Windows)
        out_bin = fp.parent / (fp.stem + (".exe" if IS_WINDOWS else ""))

        # Step 1: compile
        rc, out, err, ms, to = self._exec(
            ["g++", str(fp), "-o", str(out_bin)], fp.parent)
        if rc != 0 or to:
            return ExecResult(fp, rc, out, err, ms, to)

        # Step 2: run
        rc2, out2, err2, ms2, to2 = self._exec(
            [str(out_bin)], fp.parent)

        # Cleanup binary
        try:
            out_bin.unlink()
        except Exception:
            pass

        return ExecResult(fp, rc2, out2, err2, ms + ms2, to2)

    def _run_rust(self, fp: Path) -> ExecResult:
        """Compile with rustc (GNU toolchain on Windows) then run."""
        out_bin = fp.parent / (fp.stem + (".exe" if IS_WINDOWS else ""))

        # Use rustup to invoke the GNU toolchain explicitly on Windows
        if IS_WINDOWS:
            compile_cmd = [
                "rustup", "run", "stable-x86_64-pc-windows-gnu",
                "rustc", str(fp), "-o", str(out_bin)
            ]
        else:
            compile_cmd = ["rustc", str(fp), "-o", str(out_bin)]

        # Step 1: compile
        rc, out, err, ms, to = self._exec(compile_cmd, fp.parent)
        if rc != 0 or to:
            return ExecResult(fp, rc, out, err, ms, to)

        # Step 2: run
        rc2, out2, err2, ms2, to2 = self._exec(
            [str(out_bin)], fp.parent)

        # Cleanup binary
        try:
            out_bin.unlink()
        except Exception:
            pass

        return ExecResult(fp, rc2, out2, err2, ms + ms2, to2)

    def _run_generic(self, fp: Path) -> ExecResult:
        """Fallback: try running file directly."""
        rc, out, err, ms, to = self._exec([str(fp)], fp.parent)
        return ExecResult(fp, rc, out, err, ms, to)


# ═══════════════════════════════════════════════════════════════════
#  RICH UI HELPERS
# ═══════════════════════════════════════════════════════════════════

PHASE_ICONS   = {Phase.THINKING:"🧠", Phase.ACTING:"✍️",
                  Phase.OBSERVING:"🔍", Phase.CORRECTING:"🔧",
                  Phase.SUCCEEDED:"🎉", Phase.FAILED:"💀"}
PHASE_COLORS  = {Phase.THINKING:"bright_cyan", Phase.ACTING:"bright_magenta",
                  Phase.OBSERVING:"bright_yellow", Phase.CORRECTING:"bright_red",
                  Phase.SUCCEEDED:"bright_green", Phase.FAILED:"red"}

SYNTAX_MAP = {
    "python": "python", "javascript": "javascript",
    "bash": "bash", "java": "java",
    "cpp": "cpp", "go": "go", "rust": "rust",
}


def banner(state: AgentState) -> None:
    lang_icons = " · ".join(f"{v.icon}{v.name}" for v in LANGUAGES.values())
    art = (
        "  ╔═══════════════════════════════════════════════════════╗\n"
        "  ║         AuraCode  ·  V2 PRO  ·  The Executor          ║\n"
        "  ║              [ Ollama Local Edition ]                 ║\n"
        "  ╚═══════════════════════════════════════════════════════╝"
    )
    console.print(Panel(
        Align(Text(art, style="bold bright_cyan"), align="center"),
        border_style="bright_blue", padding=(0, 1)))

    console.print(Panel(
        f"  [bold]Model   :[/bold] [cyan]{state.model_id}[/cyan]\n"
        f"  [bold]Language:[/bold] {state.language.icon} [green]{state.language.name}[/green]\n"
        f"  [bold]Output  :[/bold] [dim]generated_code/v2/[/dim]\n\n"
        f"  [dim]{lang_icons}[/dim]",
        title="[bold]⚡ Current Configuration[/bold]",
        border_style="cyan", padding=(0, 2)))

    console.print(Panel(
        "  [bold cyan]Ctrl+L[/bold cyan]  Switch Model        "
        "[bold green]Ctrl+K[/bold green]  Switch Language\n"
        "  [bold yellow]F2[/bold yellow]     View History        "
        "[bold red]Ctrl+C[/bold red]   Exit",
        title="[bold]⌨  Hotkeys[/bold]",
        border_style="dim", padding=(0, 2)))
    console.print()


def phase_header(phase: Phase, attempt: int, max_r: int,
                 lang: Language) -> None:
    c = PHASE_COLORS[phase]
    console.print()
    console.print(Rule(
        f"[bold {c}]{PHASE_ICONS[phase]}  {phase.name}  "
        f"—  Attempt {attempt}/{max_r}  ·  "
        f"{lang.icon} {lang.name}[/bold {c}]",
        style=c))


def thought_panel(thought: str) -> None:
    if thought.strip():
        console.print(Panel(
            Text(thought[:800], style="italic bright_cyan"),
            title="[bold bright_cyan]🧠 Agent Reasoning[/bold bright_cyan]",
            border_style="bright_cyan", padding=(1, 2)))


def code_panel(code: GeneratedCode) -> None:
    name   = code.filepath.name if code.filepath else code.filename
    lexer  = SYNTAX_MAP.get(code.language.key, "text")
    console.print(Panel(
        Syntax(code.code, lexer, theme="monokai", line_numbers=True),
        title=(f"[magenta]✍️  {name}  ·  "
               f"{code.language.icon}{code.language.name}  "
               f"(attempt #{code.attempt})[/magenta]"),
        border_style="magenta", padding=(0, 1)))


def exec_panel(r: ExecResult, attempt: int) -> None:
    color = "green" if r.success else "red"
    t = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 1))
    t.add_column(style="bold white", width=14)
    t.add_column(style="bright_white")
    t.add_row("Status",
              f"[{color}]{'✅ SUCCESS' if r.success else '❌ FAILED'}[/{color}]")
    t.add_row("Code",   str(r.returncode))
    t.add_row("Error",  r.error_type if not r.success else "—")
    t.add_row("Time",   f"{r.duration_ms:.0f} ms")
    console.print(Panel(t,
        title=f"[bold]🔍 Observation · Attempt #{attempt}[/bold]",
        border_style=color))
    if r.stdout.strip():
        console.print(Panel(
            Text(r.stdout.strip()[:2000], style="bright_white"),
            title="[green]📤 OUTPUT[/green]", border_style="green"))
    if r.stderr.strip() and not r.skipped:
        console.print(Panel(
            Text(r.stderr.strip()[:2000], style="bright_red"),
            title="[red]⚠  STDERR[/red]", border_style="red"))


def show_history(history: list[dict]) -> None:
    console.print()
    if not history:
        console.print(Panel(
            "[dim]No tasks yet. Run a task first![/dim]",
            title="[bold]📜 Task History[/bold]",
            border_style="dim"))
        return
    t = Table(box=box.ROUNDED, border_style="bright_blue")
    t.add_column("#",        style="dim",          width=3)
    t.add_column("Time",     style="dim",          width=19)
    t.add_column("Task",     style="bright_white", width=38)
    t.add_column("Language", style="cyan",         width=12)
    t.add_column("Model",    style="yellow",       width=14)
    t.add_column("Status",   style="bold",         width=8)
    t.add_column("File",     style="dim",          width=22)
    for i, h in enumerate(history, 1):
        status = "[green]✅[/green]" if h["success"] else "[red]❌[/red]"
        task   = h["task"][:36] + "…" if len(h["task"]) > 36 else h["task"]
        fname  = Path(h["file"]).name if h["file"] else "—"
        t.add_row(str(i), h["timestamp"], task,
                  h["language"], h["model"][:12], status, fname)
    console.print(Panel(t,
        title=f"[bold]📜 Task History  ({len(history)} tasks)[/bold]",
        border_style="bright_blue"))


def show_model_selector(models: list[str], current: str) -> None:
    console.print()
    t = Table(box=box.ROUNDED, border_style="cyan")
    t.add_column("#",      style="dim",   width=4)
    t.add_column("Model",  style="cyan",  width=35)
    t.add_column("Active", style="green", width=8)
    for i, m in enumerate(models, 1):
        t.add_row(str(i), m, "✅ YES" if m == current else "")
    console.print(Panel(t,
        title="[bold cyan]🔄 Available Models[/bold cyan]",
        border_style="cyan"))


def show_lang_selector(current_key: str) -> None:
    console.print()
    t = Table(box=box.ROUNDED, border_style="green")
    t.add_column("#",          style="dim",    width=3)
    t.add_column("Language",   style="green",  width=14)
    t.add_column("Extension",  style="dim",    width=8)
    t.add_column("Runtime",    style="yellow", width=10)
    t.add_column("Installed",  style="bold",   width=10)
    t.add_column("Active",     style="cyan",   width=8)
    for i, (key, lang) in enumerate(LANGUAGES.items(), 1):
        installed = "✅ Yes" if is_runtime_available(lang) else "❌ No"
        active    = "✅ YES" if key == current_key else ""
        t.add_row(str(i), f"{lang.icon} {lang.name}",
                  lang.extension, lang.runtime, installed, active)
    console.print(Panel(t,
        title="[bold green]🌐 Available Languages[/bold green]",
        border_style="green"))


def success_panel(s: Session) -> None:
    elapsed = (datetime.datetime.now() - s.started_at).total_seconds()
    console.print()
    console.print(Rule(
        "[bold bright_green]🎉  MISSION ACCOMPLISHED[/bold bright_green]",
        style="green"))
    console.print(Panel(
        f"[bold]Task:[/bold]      {s.task[:80]}\n"
        f"[bold]Language:[/bold]  {s.language.icon} {s.language.name}\n"
        f"[bold]Attempts:[/bold]  {s.total_attempts}\n"
        f"[bold]Elapsed:[/bold]   {elapsed:.1f}s\n"
        f"[bold]Model:[/bold]     {s.model_id}\n"
        f"[bold]Output:[/bold]    generated_code/v2/",
        title="[bold green]✅ Session Summary[/bold green]",
        border_style="green", padding=(1, 2)))


def failure_panel(s: Session, max_r: int) -> None:
    elapsed = (datetime.datetime.now() - s.started_at).total_seconds()
    console.print()
    console.print(Rule(
        "[bold red]💀  MAX RETRIES EXCEEDED[/bold red]", style="red"))
    console.print(Panel(
        f"[bold]Task:[/bold]      {s.task[:80]}\n"
        f"[bold]Language:[/bold]  {s.language.icon} {s.language.name}\n"
        f"[bold]Attempts:[/bold]  {s.total_attempts}/{max_r}\n"
        f"[bold]Elapsed:[/bold]   {elapsed:.1f}s",
        title="[bold red]❌ Session Summary[/bold red]",
        border_style="red", padding=(1, 2)))


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════

def gen_system_prompt(lang: Language) -> str:
    extras = {
        "python":     "- Always add if __name__ == '__main__': guard\n- Use type hints\n- NEVER use input() or any function that waits for user input\n- Script must run completely on its own without any interaction",
        "javascript": "- Use modern ES6+ syntax\n- Use console.log() for all output",
        "bash":       "- Start with #!/bin/bash\n- Use echo for output",
        "java":       "- Write ONE public class\n- Class name must match the filename\n- Include public static void main(String[] args)",
        "cpp":        "- Include all needed headers (iostream, vector, etc)\n- Use int main() as entry point\n- Use cout for output",
        "go":         "- Use 'package main'\n- Import fmt\n- Use func main()",
        "rust":       "- Use fn main()\n- ONLY use Rust std library, NO extern crates\n- For random numbers: use std::time for seed-based pseudo-random, NO use rand\n- Use println! for output",
    }.get(lang.key, "")

    return f"""\
You are an expert {lang.name} code generator.
Always respond in EXACTLY this structure — no exceptions:

##FILENAME##
<name_without_extension>
##DESCRIPTION##
<one sentence what the script does>
##DEPENDENCIES##
<for Python only: pip packages comma-separated, or "none"; for all other languages: none>
##CODE##
<complete runnable {lang.name} code — no markdown fences>
##END##

Rules:
- Start with ##FILENAME## immediately — no intro text
- No ``` fences inside ##CODE##
- No text before ##FILENAME## or after ##END##
- Code must be complete and runnable
{extras}
"""


def repair_system_prompt(lang: Language) -> str:
    return f"""\
You are an expert {lang.name} debugger. A script you generated failed.
Diagnose the error and return a complete fixed version.

Always respond in EXACTLY this structure:

##THOUGHT##
<what caused the error and exactly how you will fix it>
##FILENAME##
<same filename>
##DESCRIPTION##
<one sentence>
##DEPENDENCIES##
<pip packages or: none>
##CODE##
<complete FIXED {lang.name} code — full file, never truncate>
##END##

Rules:
- Start with ##THOUGHT## immediately
- For Python ModuleNotFoundError → add the missing package to ##DEPENDENCIES##
- Write the COMPLETE fixed file
- No ``` fences inside ##CODE##
- CRITICAL: If error is TimeoutError, the script is using input() which blocks execution
  Fix by REMOVING all input() calls and replacing with hardcoded demo values
- Script must run fully automatically without any user interaction
"""


# ═══════════════════════════════════════════════════════════════════
#  FILE SAVER
# ═══════════════════════════════════════════════════════════════════

def save_code(code: GeneratedCode) -> Path:
    """
    Save generated code to a language-specific subfolder.

    Structure:
        generated_code/v2/
            python/       ← Python scripts
            javascript/   ← JS scripts
            bash/         ← Bash scripts
            java/         ← Java files
            cpp/          ← C++ files
            go/           ← Go files
            rust/         ← Rust files
    """
    import re as _re

    ext  = code.language.extension
    name = code.filename

    # Aggressively strip ALL known extensions to prevent double-ext
    for known_ext in [".py", ".js", ".sh", ".java", ".cpp", ".go", ".rs",
                      "_py", "_js", "_sh", "_java", "_cpp", "_go", "_rs"]:
        while name.endswith(known_ext):
            name = name[:-len(known_ext)]

    # Remove trailing underscores/dashes
    name = name.rstrip("_-")

    # Sanitize: only alphanumeric and underscores
    name = _re.sub(r'[^\w]', '_', name).strip('_')
    if not name:
        name = "generated_script"

    # Java: filename MUST match public class name
    if code.language.key == "java":
        m = _re.search(r'public\s+class\s+(\w+)', code.code)
        if m:
            name = m.group(1)

    # ── Save to language-specific subfolder ──────────────────────
    lang_dir = OUTPUT_DIR / code.language.key
    lang_dir.mkdir(parents=True, exist_ok=True)

    final_name = name + ext
    fp = lang_dir / final_name
    fp.write_text(code.code, encoding="utf-8")
    code.filepath = fp
    logger.info("Saved: %s", fp)
    return fp


# ═══════════════════════════════════════════════════════════════════
#  SELF-HEALING AGENT
# ═══════════════════════════════════════════════════════════════════

class AuraCodeAgentV2Pro:
    """
    V2 PRO Executor Agent with multi-language support.
    ReAct loop: THINK → ACT → OBSERVE → CORRECT → ... → SUCCEED/FAIL
    """

    def __init__(self, state: AgentState, max_retries: int = MAX_RETRIES):
        self.state       = state
        self.max_retries = max_retries
        self._env        = Env()
        self._log        = logging.getLogger("AuraCode.V2Pro.Agent")

    def _llm(self, sys_prompt: str, messages: list[dict],
             spinner: str) -> str:
        """Call Ollama with spinner. Returns raw response text."""
        with Progress(
            SpinnerColumn(spinner_name="dots2", style="bright_cyan"),
            TextColumn(f"[bright_cyan]  {spinner}[/bright_cyan]"),
            TimeElapsedColumn(),
            transient=True, console=console,
        ) as p:
            p.add_task("l", total=None)
            resp = self.state.get_client().chat.completions.create(
                model=self.state.model_id,
                messages=[{"role": "system", "content": sys_prompt}] + messages,
                temperature=0.1,
            )
        raw = resp.choices[0].message.content
        self._log.debug("LLM response (%d chars)", len(raw))
        return raw

    def _initial(self, task: str) -> GeneratedCode:
        """Generate code from scratch."""
        lang = self.state.language
        raw  = self._llm(
            gen_system_prompt(lang),
            [{"role": "user",
              "content": f"Write a {lang.name} script that does:\n\n{task}"}],
            f"Generating {lang.icon} {lang.name} code…",
        )
        if not raw.strip():
            raise ValueError("Model returned empty response. Try again.")
        parsed = parse_llm_response(raw, attempt=1)
        return GeneratedCode.from_parsed(parsed, lang, attempt=1)

    def _repair(self, task: str, broken: GeneratedCode,
                result: ExecResult, attempt: int,
                history: list[dict]) -> GeneratedCode:
        """Diagnose and fix failing code."""
        lang = self.state.language
        msg  = (
            f"TASK: {task}\n\n"
            f"ERROR TYPE: {result.error_type}\n\n"
            f"STDERR:\n{result.stderr[:2000]}\n\n"
            f"STDOUT:\n{result.stdout[:300]}\n\n"
            f"BROKEN CODE:\n{broken.code}\n\n"
            "Fix the bug and return the complete corrected script."
        )
        msgs   = history + [{"role": "user", "content": msg}]
        raw    = self._llm(
            repair_system_prompt(lang), msgs,
            f"Self-healing · attempt #{attempt}…")
        if not raw.strip():
            raise ValueError("Model returned empty repair response.")
        parsed = parse_llm_response(raw, attempt=attempt)
        return GeneratedCode.from_parsed(parsed, lang, attempt=attempt)

    def run(self, task: str) -> Session:
        """Execute the full ReAct self-healing loop."""
        lang    = self.state.language
        session = Session(task=task, language=lang,
                          model_id=self.state.model_id)
        history: list[dict] = []
        current: Optional[GeneratedCode] = None
        result:  Optional[ExecResult]    = None

        self._log.info("Session: lang=%s model=%s task=%.80s",
                       lang.name, self.state.model_id, task)

        for attempt in range(1, self.max_retries + 1):

            # ── THINK ────────────────────────────────────────────
            phase_header(Phase.THINKING, attempt, self.max_retries, lang)
            try:
                if attempt == 1:
                    current = self._initial(task)
                else:
                    current = self._repair(
                        task, current, result, attempt, history)
                    thought_panel(current.thought)
            except ValueError as exc:
                self._log.error("Parse failed attempt %d: %s", attempt, exc)
                console.print(Panel(str(exc)[:400],
                    title="[red]⚠ Parse Error — retrying[/red]",
                    border_style="red"))
                if current is None:
                    session.final_phase = Phase.FAILED
                    return session
                history.append({"role": "user",
                    "content": "Your response could not be parsed. "
                               "Please strictly follow the output format."})
                continue

            history.append({"role": "assistant",
                "content": (f"##THOUGHT##\n{current.thought}\n"
                            f"##FILENAME##\n{current.filename}\n"
                            f"##CODE##\n{current.code}\n##END##")})

            # ── ACT ──────────────────────────────────────────────
            phase_header(Phase.ACTING, attempt, self.max_retries, lang)
            fp = save_code(current)
            code_panel(current)
            console.print(f"  [dim]→ Saved: {fp}[/dim]")

            if current.has_deps:
                self._env.install_python_deps(current.dependencies)

            # ── OBSERVE ──────────────────────────────────────────
            phase_header(Phase.OBSERVING, attempt, self.max_retries, lang)
            with Progress(
                SpinnerColumn(style="yellow"),
                TextColumn(
                    f"[yellow]  Running {lang.icon} {lang.name}…[/yellow]"),
                TimeElapsedColumn(),
                transient=True, console=console,
            ) as p:
                p.add_task("r", total=None)
                result = self._env.run(fp, lang)

            exec_panel(result, attempt)
            session.log(attempt, current, result)

            # ── SUCCESS ───────────────────────────────────────────
            if result.success:
                session.final_phase = Phase.SUCCEEDED
                success_panel(session)
                self._save_session(session)
                self.state.add_history(
                    task, lang.name, self.state.model_id,
                    True, str(fp))
                return session

            # ── CORRECT ──────────────────────────────────────────
            if attempt < self.max_retries:
                phase_header(
                    Phase.CORRECTING, attempt, self.max_retries, lang)
                console.print(Panel(
                    f"[yellow]Error:[/yellow] "
                    f"[bold red]{result.error_type}[/bold red]\n"
                    "[yellow]Feeding traceback to LLM for repair…[/yellow]",
                    border_style="yellow"))
                history.append({"role": "user",
                    "content": (
                        f"Attempt #{attempt} failed.\n"
                        f"Error: {result.error_type}\n"
                        f"STDERR:\n{result.stderr[:1500]}")})

        # ── FAILED ───────────────────────────────────────────────
        session.final_phase = Phase.FAILED
        failure_panel(session, self.max_retries)
        self._save_session(session)
        self.state.add_history(
            task, lang.name, self.state.model_id, False,
            str(current.filepath) if current and current.filepath else "")
        return session

    def _save_session(self, s: Session) -> None:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = LOG_DIR / f"v2pro_{ts}.json"
        path.write_text(s.to_json(), encoding="utf-8")
        console.print(f"  [dim]📝 Session log → {path}[/dim]")


# ═══════════════════════════════════════════════════════════════════
#  INTERACTIVE CLI WITH HOTKEYS
# ═══════════════════════════════════════════════════════════════════

def build_keybindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-l")
    def _(event):
        event.app.exit(result="__SWITCH_MODEL__")

    @kb.add("c-k")
    def _(event):
        event.app.exit(result="__SWITCH_LANG__")

    @kb.add("f2")
    def _(event):
        event.app.exit(result="__SHOW_HISTORY__")

    return kb


def run_cli(state: AgentState) -> None:
    """Main interactive loop with hotkey support."""
    agent   = AuraCodeAgentV2Pro(state)
    kb      = build_keybindings()
    session = PromptSession(
        history=InMemoryHistory(),
        key_bindings=kb,
        style=Style.from_dict({
            "bottom-toolbar": "bg:#1a1a2e fg:#888888",
        }),
    )

    banner(state)

    while True:
        try:
            toolbar = HTML(
                f" <b>Model:</b> {state.model_id}  "
                f"<b>Lang:</b> {state.language.icon}{state.language.name}  "
                f"<b>History:</b> {len(state.history)} tasks  "
                "│  <b>Ctrl+L</b> Model  "
                "<b>Ctrl+K</b> Language  "
                "<b>F2</b> History"
            )
            task = session.prompt(
                HTML("<ansicyan><b>► </b></ansicyan>"
                     "<ansigreen>Describe your script</ansigreen>"
                     "<ansicyan><b>: </b></ansicyan>"),
                bottom_toolbar=toolbar,
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break

        # ── Hotkey handlers ───────────────────────────────────────

        if task == "__SWITCH_MODEL__":
            models = get_ollama_models()
            if not models:
                console.print(
                    "[red]Could not fetch models. Is Ollama running?[/red]")
                continue
            state.available_models = models
            show_model_selector(models, state.model_id)
            console.print()
            try:
                choice = console.input(
                    f"[bold cyan]Enter number (1-{len(models)})"
                    " or model name: [/bold cyan]"
                ).strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        state.set_model(models[idx])
                        console.print(
                            f"[green]✓ Model → [bold]{state.model_id}[/bold][/green]")
                    else:
                        console.print("[red]Invalid number.[/red]")
                elif choice:
                    state.set_model(choice)
                    console.print(
                        f"[green]✓ Model → [bold]{state.model_id}[/bold][/green]")
            except KeyboardInterrupt:
                pass
            continue

        if task == "__SWITCH_LANG__":
            show_lang_selector(state.language.key)
            console.print()
            try:
                choice = console.input(
                    f"[bold green]Enter number (1-{len(LANGUAGE_LIST)})"
                    " or language name: [/bold green]"
                ).strip().lower()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(LANGUAGE_LIST):
                        state.language = LANGUAGES[LANGUAGE_LIST[idx]]
                        console.print(
                            f"[green]✓ Language → "
                            f"[bold]{state.language.icon} "
                            f"{state.language.name}[/bold][/green]")
                    else:
                        console.print("[red]Invalid number.[/red]")
                elif choice in LANGUAGES:
                    state.language = LANGUAGES[choice]
                    console.print(
                        f"[green]✓ Language → "
                        f"[bold]{state.language.icon} "
                        f"{state.language.name}[/bold][/green]")
                else:
                    console.print(
                        f"[red]Unknown: {choice}. "
                        f"Options: {', '.join(LANGUAGE_LIST)}[/red]")
            except KeyboardInterrupt:
                pass
            continue

        if task == "__SHOW_HISTORY__":
            show_history(state.history)
            continue

        # ── Normal task ───────────────────────────────────────────
        if not task or not task.strip():
            continue
        if task.strip().lower() in {"exit", "quit", "q"}:
            console.print("[dim]AuraCode shutting down. Goodbye.[/dim]")
            break

        try:
            agent.run(task.strip())
        except Exception as exc:
            logger.exception("Unexpected: %s", exc)
            console.print(f"[bold red]Error:[/bold red] {exc}")
            if "connection" in str(exc).lower():
                console.print(
                    "[yellow]→ Make sure Ollama is running[/yellow]")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    state  = AgentState()
    models = get_ollama_models()
    if models:
        state.available_models = models
        if state.model_id not in models:
            state.set_model(models[0])
    run_cli(state)


if __name__ == "__main__":
    main()