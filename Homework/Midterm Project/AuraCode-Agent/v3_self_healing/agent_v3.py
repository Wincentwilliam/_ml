"""
AuraCode-Agent · Version 3.0 PRO · The Self-Healing Agent
Ollama Local Edition — Auto Language Detection + Website Support

Features:
  · AUTO language detection from prompt — no need to select manually
  · Support: Python, JavaScript, Bash, Java, C++, Go, Rust, HTML/Website
  · Asks user if language is ambiguous
  · Ctrl+L  — switch Ollama model on-the-fly
  · F2      — view full task history
  · Ctrl+C  — exit
  · Self-healing ReAct loop: Think → Act → Observe → Correct
  · Language-specific output subfolders

Author: AuraCode-Agent V3 PRO
"""

from __future__ import annotations

import sys
import json
import os
import re
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
OUTPUT_DIR      = BASE_DIR / "generated_code" / "v3"
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
logger  = logging.getLogger("AuraCode.V3Pro")
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
    runtime:   str
    runnable:  bool = True   # False = just save, no execute (e.g. HTML)


LANGUAGES: dict[str, Language] = {
    "python":     Language("Python",     "python",     ".py",   "🐍", "python"),
    "javascript": Language("JavaScript", "javascript", ".js",   "🟨", "node"),
    "bash":       Language("Bash",       "bash",       ".sh",   "🔧", "bash"),
    "java":       Language("Java",       "java",       ".java", "☕", "javac"),
    "cpp":        Language("C++",        "cpp",        ".cpp",  "⚡", "g++"),
    "go":         Language("Go",         "go",         ".go",   "🐹", "go"),
    "rust":       Language("Rust",       "rust",       ".rs",   "🦀", "rustc"),
    "html":       Language("HTML/Web",   "html",       ".html", "🌐", "",     runnable=False),
    "fullstack":  Language("Full-Stack",  "fullstack",  "",      "🚀", "python", runnable=False),
}

LANGUAGE_LIST = list(LANGUAGES.keys())


def is_runtime_available(lang: Language) -> bool:
    if not lang.runtime:
        return True   # HTML needs no runtime
    return shutil.which(lang.runtime) is not None


# ═══════════════════════════════════════════════════════════════════
#  AUTO LANGUAGE DETECTOR
# ═══════════════════════════════════════════════════════════════════

# Keywords that strongly signal a specific language
LANG_KEYWORDS: dict[str, list[str]] = {
    "fullstack":  ["fullstack", "full stack", "full-stack", "website with backend",
                   "fastapi", "flask website", "web app with ai", "website with chat",
                   "website dengan ai", "web dengan chatbot", "web app", "webapp",
                   "website dengan backend", "coffee shop website", "restaurant website",
                   "portfolio website", "toko online", "web toko",
                   "dengan chatbot", "with chatbot", "dengan ai chat",
                   "with ai chat", "ai barista", "chatbot barista",
                   "website dengan chat", "site dengan ai"],
    "html":       ["website", "web", "html", "webpage", "landing page",
                   "coffee shop site", "homepage", "css", "frontend",
                   "portfolio site", "blog site", "ui", "web page"],
    "python":     ["python", "py", "flask", "django", "pandas", "numpy",
                   "machine learning", "data science", "matplotlib"],
    "javascript": ["javascript", "js", "node", "nodejs", "react", "vue",
                   "express", "typescript"],
    "bash":       ["bash", "shell", "sh script", "shell script", "linux command",
                   "terminal script"],
    "java":       ["java", "spring", "maven", "gradle", "jvm"],
    "cpp":        ["c++", "cpp", "c plus plus", "c++ program", "cplusplus",
                   "using c++", "in c++", "with c++"],
    "go":         ["golang", "go program", "go script", "in go", "using go",
                   "with go", "go lang", "go programming", " go ", "go multiplication",
                   "go fibonacci", "go calculator", "go sorting", "go table"],
    "rust":       ["rust", "rs program", "cargo"],
}

# Ambiguous tasks that could be any language
AMBIGUOUS_KEYWORDS = [
    "calculator", "kalkulator", "todo", "to-do", "clock", "timer",
    "game", "quiz", "converter", "buat program", "create a program",
]


def detect_language(task: str) -> Optional[str]:
    """
    Detect the intended programming language from the task description.

    Returns:
        Language key (e.g. 'python', 'html') if confident,
        None if ambiguous (will ask user).
    """
    task_lower = " " + task.lower() + " "  # pad for word-boundary matching

    # Check for explicit language mentions — highest priority
    for lang_key, keywords in LANG_KEYWORDS.items():
        for kw in keywords:
            if kw in task_lower:
                logger.info("Auto-detected language: %s (keyword: %s)", lang_key, kw)
                return lang_key

    # If no explicit language found, check if it's clearly ambiguous
    for amb in AMBIGUOUS_KEYWORDS:
        if amb in task_lower:
            logger.info("Ambiguous task detected — will ask user")
            return None

    # Default: if mentions "script" or "program" without language, ask
    if any(w in task_lower for w in ["script", "program", "code", "buat", "create", "make"]):
        return None

    # Fallback: Python (most common)
    logger.info("No language detected, defaulting to Python")
    return "python"


def ask_language_choice() -> Language:
    """Ask the user to choose a language when task is ambiguous."""
    console.print()
    console.print(Panel(
        "[yellow]🤔 Hmm, bahasa apa yang mau dipakai?[/yellow]\n\n"
        + "\n".join(
            f"  [bold]{i}[/bold]. {lang.icon} {lang.name}"
            for i, (key, lang) in enumerate(LANGUAGES.items(), 1)
        ),
        title="[bold yellow]Pilih Bahasa[/bold yellow]",
        border_style="yellow",
    ))
    console.print()

    while True:
        choice = console.input(
            "[bold yellow]► Masukkan nomor atau nama bahasa: [/bold yellow]"
        ).strip().lower()

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(LANGUAGE_LIST):
                return LANGUAGES[LANGUAGE_LIST[idx]]
        elif choice in LANGUAGES:
            return LANGUAGES[choice]
        # fuzzy match
        for key, lang in LANGUAGES.items():
            if choice in lang.name.lower():
                return lang

        console.print(f"[red]Tidak valid. Ketik angka 1-{len(LANGUAGES)} atau nama bahasa.[/red]")


# ═══════════════════════════════════════════════════════════════════
#  OLLAMA MODEL DETECTION
# ═══════════════════════════════════════════════════════════════════

def get_ollama_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []
        return [line.split()[0] for line in result.stdout.strip().splitlines()[1:]
                if line.split()]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════
#  AGENT STATE
# ═══════════════════════════════════════════════════════════════════

class AgentState:
    def __init__(self):
        self.model_id:         str        = "qwen3.5:9b"
        self.history:          list[dict] = []
        self.available_models: list[str]  = []
        self._client: Optional[OpenAI]    = None

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
        self._client  = None

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
        if self.skipped:   return "Saved (no execution needed)"
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
            "attempt": n, "filename": code.filename,
            "thought": code.thought, "success": r.success,
            "error_type": r.error_type, "duration_ms": r.duration_ms,
        })
        self.total_attempts = n

    def to_json(self) -> str:
        return json.dumps({
            "task": self.task, "language": self.language.name,
            "model": self.model_id,
            "started_at": self.started_at.isoformat(),
            "final_phase": self.final_phase.name,
            "total_attempts": self.total_attempts,
            "attempts": self.attempts,
        }, indent=2)


# ═══════════════════════════════════════════════════════════════════
#  EXECUTION ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════

class Env:
    def __init__(self, timeout: int = EXEC_TIMEOUT):
        self.timeout   = timeout
        self.python    = sys.executable
        self._done:    set[str] = set()

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

    def run(self, fp: Path, lang: Language) -> ExecResult:
        # HTML/Web: just save, open in browser prompt
        if not lang.runnable:
            console.print(Panel(
                f"[green]🌐 Website generated![/green]\n"
                f"File saved: [bold]{fp}[/bold]\n\n"
                f"[dim]Open in browser: start {fp}[/dim]",
                border_style="green"))
            return ExecResult(fp, 0, f"Website saved to {fp}", "", 0.0, skipped=True)

        if not is_runtime_available(lang):
            console.print(Panel(
                f"[yellow]⚠  {lang.name} runtime not found.\n"
                f"File saved: [bold]{fp}[/bold][/yellow]",
                border_style="yellow"))
            return ExecResult(fp, 0, "", "", 0.0, skipped=True)

        dispatch = {
            "python":     self._run_python,
            "javascript": self._run_javascript,
            "bash":       self._run_bash,
            "java":       self._run_java,
            "cpp":        self._run_cpp,
            "go":         self._run_go,
            "rust":       self._run_rust,
        }
        runner = dispatch.get(lang.key, self._run_python)
        return runner(fp)

    def _exec(self, cmd: list[str], cwd: Path) -> tuple:
        start = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, cwd=cwd,
                env={**os.environ},
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
        import tempfile, shutil as _sh
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_fp  = tmp_dir / "script.sh"
        _sh.copy2(str(fp), str(tmp_fp))
        if IS_WINDOWS:
            posix = tmp_fp.as_posix()
            if len(posix) > 2 and posix[1] == ":":
                posix = "/" + posix[0].lower() + posix[2:]
            bash_exe = shutil.which("bash") or "bash"
            is_wsl   = "system32" in bash_exe.lower() or "windowsapps" in bash_exe.lower()
            if is_wsl:
                script = fp.read_text(encoding="utf-8").replace("\r\n", "\n")
                start  = time.perf_counter()
                timed_out = False
                try:
                    proc = subprocess.run(
                        ["bash"], input=script,
                        capture_output=True, text=True,
                        timeout=self.timeout, env={**os.environ})
                    rc, out, err = proc.returncode, proc.stdout, proc.stderr
                except subprocess.TimeoutExpired:
                    timed_out = True
                    rc, out, err = -1, "", f"[Timeout] Exceeded {self.timeout}s"
                except Exception as exc:
                    rc, out, err = -1, "", str(exc)
                ms = (time.perf_counter() - start) * 1000
                try: _sh.rmtree(str(tmp_dir))
                except: pass
                return ExecResult(fp, rc, out, err, ms, timed_out)
            else:
                rc, out, err, ms, to = self._exec(["bash", posix], tmp_dir)
        else:
            rc, out, err, ms, to = self._exec(["bash", str(tmp_fp)], tmp_dir)
        try: _sh.rmtree(str(tmp_dir))
        except: pass
        return ExecResult(fp, rc, out, err, ms, to)

    def _run_java(self, fp: Path) -> ExecResult:
        rc, out, err, ms, to = self._exec(["javac", str(fp)], fp.parent)
        if rc != 0 or to:
            return ExecResult(fp, rc, out, err, ms, to)
        rc2, out2, err2, ms2, to2 = self._exec(
            ["java", "-cp", str(fp.parent), fp.stem], fp.parent)
        return ExecResult(fp, rc2, out2, err2, ms + ms2, to2)

    def _run_cpp(self, fp: Path) -> ExecResult:
        out_bin = fp.parent / (fp.stem + (".exe" if IS_WINDOWS else ""))
        rc, out, err, ms, to = self._exec(
            ["g++", str(fp), "-o", str(out_bin)], fp.parent)
        if rc != 0 or to:
            return ExecResult(fp, rc, out, err, ms, to)
        rc2, out2, err2, ms2, to2 = self._exec([str(out_bin)], fp.parent)
        try: out_bin.unlink()
        except: pass
        return ExecResult(fp, rc2, out2, err2, ms + ms2, to2)

    def _run_go(self, fp: Path) -> ExecResult:
        rc, out, err, ms, to = self._exec(["go", "run", str(fp)], fp.parent)
        return ExecResult(fp, rc, out, err, ms, to)

    def _run_rust(self, fp: Path) -> ExecResult:
        out_bin = fp.parent / (fp.stem + (".exe" if IS_WINDOWS else ""))
        if IS_WINDOWS:
            compile_cmd = ["rustup", "run", "stable-x86_64-pc-windows-gnu",
                           "rustc", str(fp), "-o", str(out_bin)]
        else:
            compile_cmd = ["rustc", str(fp), "-o", str(out_bin)]
        rc, out, err, ms, to = self._exec(compile_cmd, fp.parent)
        if rc != 0 or to:
            return ExecResult(fp, rc, out, err, ms, to)
        rc2, out2, err2, ms2, to2 = self._exec([str(out_bin)], fp.parent)
        try: out_bin.unlink()
        except: pass
        return ExecResult(fp, rc2, out2, err2, ms + ms2, to2)


# ═══════════════════════════════════════════════════════════════════
#  FILE SAVER
# ═══════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════
#  FULLSTACK PROJECT GENERATOR
#  Generates a complete multi-file web project in one shot
# ═══════════════════════════════════════════════════════════════════

FULLSTACK_SYSTEM_PROMPT = """
You are an elite full-stack web developer. Generate a stunning, FULLY FUNCTIONAL multi-file web project.

OUTPUT CONTRACT — follow EXACTLY:

##PROJECT_NAME##
<snake_case_name>
##PROJECT_DESCRIPTION##
<one sentence>

##FILE: index.html##
<COMPLETE HTML — see requirements below>

##FILE: style.css##
<COMPLETE CSS — see requirements below>

##FILE: script.js##
<COMPLETE JavaScript — see requirements below>

##FILE: app.py##
<COMPLETE FastAPI backend — see requirements below>

##FILE: requirements.txt##
fastapi
uvicorn[standard]
openai
python-multipart
##END##

════════════════════════════════════════════════
INDEX.HTML REQUIREMENTS:
════════════════════════════════════════════════
1. NAVIGATION: Logo + links (Home, Menu, About, Contact). Smooth scroll.
2. HERO SECTION:
   - Full-screen background using real Unsplash photo URL:
     <img src="https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?w=1920&q=80" />
     (use relevant unsplash photo for the business type)
   - Overlay with title + subtitle + CTA button
3. MENU/PRODUCTS SECTION:
   - Grid of 6 items minimum with real Unsplash photos
   - Each item: photo, name, description, price, "Order Now" button
   - "Order Now" triggers fake payment modal
4. ABOUT SECTION: Story, values, team photo from Unsplash
5. CONTACT SECTION: Address, phone, email (fake but realistic)
6. AI CHAT WIDGET:
   - Floating button bottom-right (💬 icon)
   - Click opens chat panel sliding from right
   - Chat shows bot avatar + user avatar
   - Messages with timestamps
   - Typing indicator animation (three dots)
   - Sends POST to /api/chat and streams response
7. PAYMENT MODAL:
   - Appears when "Order Now" clicked
   - Shows: item name, price, fake card form (name, card number, expiry, CVV)
   - "Pay Now" button → 2 second loading → success screen with confetti
   - "Order #XXXXX confirmed! Estimated: 15 minutes ☕"
8. FOOTER: Links, social icons, copyright
9. Include <link rel="stylesheet" href="style.css"> and <script src="script.js">
10. DO NOT embed CSS/JS in HTML — use external files

════════════════════════════════════════════════
STYLE.CSS REQUIREMENTS:
════════════════════════════════════════════════
- CSS variables for color scheme (warm coffee tones or theme-appropriate)
- Smooth hover animations on all cards and buttons
- Floating chat button pulse animation
- Modal overlay with blur backdrop
- Mobile responsive (media queries)
- Smooth scroll behavior
- Card hover: translateY(-8px) + shadow
- Loading spinner animation
- Confetti animation (CSS keyframes)
- Chat panel slide-in animation from right

════════════════════════════════════════════════
SCRIPT.JS REQUIREMENTS:
════════════════════════════════════════════════
1. NAVIGATION: Highlight active section on scroll. Mobile hamburger menu.
2. CHAT WIDGET:
   - toggleChat() opens/closes chat panel
   - sendMessage() reads input, displays user msg, calls /api/chat
   - Shows typing indicator while waiting
   - Displays AI response with typing effect
   - Handles errors gracefully
3. PAYMENT MODAL:
   - openPayment(itemName, price) shows modal
   - Auto-format card number (groups of 4)
   - Auto-format expiry (MM/YY)
   - processPayment() → loading state → success with order number
   - generateOrderId() returns random 5-digit number
4. SMOOTH SCROLL for nav links
5. INTERSECTION OBSERVER for scroll animations
6. All buttons must work — no dead buttons

════════════════════════════════════════════════
APP.PY REQUIREMENTS:
════════════════════════════════════════════════
- FastAPI app
- StaticFiles mount: app.mount("/static", StaticFiles(directory="."), name="static")
- GET "/" serves index.html using FileResponse
- POST /api/chat:
  - Body: {"message": str, "history": list}
  - Calls Ollama at http://localhost:11434/v1 using openai library
  - Model: first available from ["qwen3.5:9b", "llama3.2:latest", "gemma4:31b-cloud"]
  - System prompt: "You are a helpful AI assistant for [business name]. Be friendly, concise, and helpful. Answer questions about the menu, hours, location, and ordering."
  - Returns {"response": str}
- CORS: allow all origins for local dev
- Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
- if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)

════════════════════════════════════════════════
CRITICAL RULES:
════════════════════════════════════════════════
- ALL files must be COMPLETE — no placeholders, no "TODO", no "..."
- Use REAL Unsplash URLs with ?w=800&q=80 for cards, ?w=1920&q=80 for hero
- Alt text on all images
- index.html links to style.css and script.js (NOT embedded)
- Every single button must do something
- Chat must connect to /api/chat backend
- Payment modal must show success with order number
- Start each file immediately after ##FILE: filename## with no intro
"""


# ═══════════════════════════════════════════════════════════════════
#  FULLSTACK PROJECT GENERATOR  — File-by-file for maximum quality
# ═══════════════════════════════════════════════════════════════════

# Each file gets its own focused prompt so the model is not overwhelmed
FILE_PROMPTS = {
    "index.html": lambda project_name, desc, task: f"""
You are an expert HTML/CSS developer. Create a COMPLETE, BEAUTIFUL, PROFESSIONAL HTML file.

Project: {project_name}
Description: {desc}
Business context: {task}

REQUIREMENTS:
1. DOCTYPE html, meta charset, viewport
2. Google Fonts link (Playfair Display + Inter)
3. <link rel="stylesheet" href="style.css"> — NO embedded CSS
4. NAVIGATION: logo + nav links (smooth scroll to sections)
5. HERO SECTION: full-screen with background image from Unsplash
   Use: <img src="https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?w=1920&q=80" class="hero-bg">
   Pick a RELEVANT Unsplash photo URL for this business type
6. CONTENT SECTIONS: at least 4 sections (hero, products/services, about, contact)
7. PRODUCTS/MENU: grid of 6+ cards, each with:
   - Real Unsplash image with onerror fallback
   - Name, description, price
   - <button class="btn-order" onclick="openPayment('Item Name', price)">Order Now</button>
8. PAYMENT MODAL: id="paymentModal" with form fields and id="modalSuccess"
9. AI CHAT WIDGET: floating button + panel id="chatPanel" with input and send button
10. FOOTER: links, social icons, copyright
11. <script src="script.js"> at bottom — NO embedded JS
12. ALL images must have onerror="this.src='https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=800&q=80'"

OUTPUT: just the complete HTML code, nothing else. No explanation, no markdown fences.
""",

    "style.css": lambda project_name, desc, task: f"""
You are an expert CSS developer. Create COMPLETE, PROFESSIONAL CSS for a {desc} website.

Project: {project_name}
Business: {task}

REQUIREMENTS:
1. CSS variables for colors (:root {{ --primary, --secondary, --accent, --bg, etc }})
2. Choose a color scheme appropriate for this business
3. Navbar: fixed top, transparent → solid on scroll (.scrolled class)
4. Hero: full-screen, overlay, centered content
5. Cards: hover effect translateY(-8px) + shadow
6. Tabs: .tab-btn active state
7. Payment modal: .modal-overlay fixed fullscreen, .modal centered card
8. .modal-overlay.active {{ display: flex }}
9. Success screen: .modal-success with animation
10. Chat panel: .chat-panel fixed bottom-right, .chat-panel.open visible
11. Chat toggle button: pulse animation
12. Typing indicator: .typing-dot bounce animation  
13. Responsive: @media (max-width: 768px)
14. Smooth transitions on all interactive elements
15. Professional typography with Playfair Display for headings

OUTPUT: just the complete CSS code, nothing else. No explanation, no markdown fences.
""",

    "script.js": lambda project_name, desc, task: f"""
You are an expert JavaScript developer. Create COMPLETE, WORKING JavaScript for a {desc} website.

Project: {project_name}

REQUIREMENTS — implement ALL of these functions:

1. initNavbar(): scroll event adds .scrolled to navbar, highlights active nav link
2. initTabs(): .tab-btn clicks show/hide .menu-card elements by data-tab-content
3. initScrollAnimations(): IntersectionObserver adds .visible to .fade-in elements
4. toggleChat(): toggles .chat-panel .open class, toggles chat-icon/close-icon visibility
5. sendMessage(): 
   - reads #chatInput value
   - calls addMessage('user', text)
   - shows #chatTyping
   - fetches POST /api/chat with JSON body: {{message: text, history: chatHistory}}
   - on response: hides typing, calls addMessage('bot', data.response)
   - handles errors with friendly fallback message
6. addMessage(role, text): creates .chat-msg div with .msg-bubble and .msg-time
7. openPayment(itemName, price): shows #paymentModal, sets #orderSummary content
8. closePayment(): hides #paymentModal
9. processPayment(event): e.preventDefault(), shows loading 2s, then shows #modalSuccess with random order number
10. spawnConfetti(): creates 20+ colored confetti pieces in #confetti div
11. formatCard(input): formats card number as "XXXX XXXX XXXX XXXX"
12. formatExpiry(input): formats as "MM/YY"
13. animateCounter(el): animates stat numbers from 0 to data-target value

IMPORTANT:
- Use 'use strict'
- Keep chatHistory array, push {{role, content}} after each message
- Modal closes when clicking overlay (id="paymentModal")
- DOMContentLoaded initializes everything

OUTPUT: just the complete JavaScript code, nothing else. No explanation, no markdown fences.
""",

    "app.py": lambda project_name, desc, task: f"""
You are an expert Python/FastAPI developer. Create a COMPLETE FastAPI backend.

Project: {project_name}
Description: {desc}

REQUIREMENTS:
1. Import: fastapi, uvicorn, StaticFiles, FileResponse, JSONResponse, CORSMiddleware, OpenAI, BaseModel
2. Mount static files: app.mount("/static", StaticFiles(directory="."), name="static")  
3. GET "/" returns FileResponse("index.html")
4. GET "/health" returns {{"status": "ok"}}
5. POST "/api/chat":
   - Body: {{message: str, history: list}}
   - Uses openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=60)
   - System prompt: "You are a helpful AI assistant for {project_name}, a {desc}. Be friendly, helpful, and concise."
   - Builds messages list: system + last 8 history items + user message
   - Calls ollama.chat.completions.create(model="llama3.2:latest", messages=messages, temperature=0.7, max_tokens=250)
   - Returns JSONResponse({{"response": reply}})
   - Has try/except that returns friendly fallback message on error
6. CORS: allow_origins=["*"]
7. if __name__ == "__main__": uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

OUTPUT: just the complete Python code, nothing else. No explanation, no markdown fences.
""",

    "requirements.txt": lambda project_name, desc, task: """fastapi
uvicorn[standard]
openai
python-multipart
""",
}


def generate_fullstack_project(state: AgentState, task: str) -> tuple:
    """
    Generate a complete multi-file full-stack project FILE BY FILE.
    Each file has its own focused prompt for maximum quality.
    Returns (project_name, files_dict).
    """
    import re as _re

    # Step 1: Extract project name and description from task
    console.print(Panel(
        f"[bold cyan]🚀 Full-Stack Generator[/bold cyan]\n"
        f"[dim]Generating 5 files one by one for maximum quality[/dim]",
        border_style="cyan"))

    # Get project name from LLM
    name_prompt = f"Given this project request: '{task}'\nRespond with ONLY a short snake_case project name (e.g. 'coffee_shop' or 'restaurant_website'). Nothing else."
    try:
        name_resp = state.get_client().chat.completions.create(
            model=state.model_id,
            messages=[{"role": "user", "content": name_prompt}],
            temperature=0.1, max_tokens=20,
        )
        raw_name    = name_resp.choices[0].message.content.strip()
        project_name = _re.sub(r'[^\w]', '_', raw_name.lower()).strip('_') or "my_project"
    except Exception:
        project_name = "web_project"

    desc = task[:80]
    console.print(f"  [green]✓[/green] Project name: [bold]{project_name}[/bold]")

    # Step 2: Generate each file with its own focused prompt
    files = {}
    file_order = ["index.html", "style.css", "script.js", "app.py", "requirements.txt"]

    for filename in file_order:
        if filename == "requirements.txt":
            files[filename] = FILE_PROMPTS[filename](project_name, desc, task)
            console.print(f"  [green]✓[/green] {filename} (template)")
            continue

        console.print()
        console.print(Rule(f"[cyan]Generating {filename}[/cyan]", style="cyan"))

        prompt = FILE_PROMPTS[filename](project_name, desc, task)

        with Progress(
            SpinnerColumn(spinner_name="dots2", style="bright_cyan"),
            TextColumn(f"[bright_cyan]  Generating {filename}…[/bright_cyan]"),
            TimeElapsedColumn(), transient=True, console=console,
        ) as p:
            p.add_task("l", total=None)
            try:
                resp = state.get_client().chat.completions.create(
                    model=state.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4096,
                )
                raw = resp.choices[0].message.content.strip()

                # Strip any markdown fences
                raw = _re.sub(r'^```[\w]*\n?', '', raw, flags=_re.MULTILINE)
                raw = _re.sub(r'\n?```\s*$', '', raw, flags=_re.MULTILINE)
                raw = raw.strip()

                # Strip <think> tags (qwen models)
                raw = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL).strip()

                if raw:
                    files[filename] = raw
                    console.print(f"  [green]✓[/green] {filename}: {len(raw):,} chars")
                else:
                    console.print(f"  [red]✗[/red] {filename}: empty response")

            except Exception as exc:
                console.print(f"  [red]✗[/red] {filename}: {exc}")

    return project_name, files


def save_fullstack_project(project_name: str, files: dict) -> Path:
    """Save all project files to generated_code/v3/fullstack/<project_name>/"""
    project_dir = OUTPUT_DIR / "fullstack" / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(Rule("[bold green]Saving Project Files[/bold green]", style="green"))

    saved = []
    for filename, content in files.items():
        fp = project_dir / filename
        fp.write_text(content, encoding="utf-8")
        saved.append(fp)
        size = len(content)
        console.print(f"  [green]✓[/green] [bold]{filename}[/bold] ({size:,} bytes)")

    console.print(Panel(
        f"[bold]Project:[/bold]   {project_name}\n"
        f"[bold]Files:[/bold]     {len(saved)} generated\n"
        f"[bold]Location:[/bold]  generated_code/v3/fullstack/{project_name}/\n\n"
        f"[bold cyan]To run:[/bold cyan]\n"
        f"  1. cd generated_code\\v3\\fullstack\\{project_name}\n"
        f"  2. pip install -r requirements.txt\n"
        f"  3. python app.py\n"
        f"  4. Open [bold]http://localhost:8000[/bold] in browser\n\n"
        f"[yellow]⚡ Keep Ollama running for AI chat![/yellow]",
        title="[bold green]🚀 Project Ready![/bold green]",
        border_style="green", padding=(1, 2)))

    return project_dir


def run_fullstack(state: AgentState, task: str) -> None:
    """Full pipeline for fullstack project generation."""
    try:
        project_name, files = generate_fullstack_project(state, task)

        if not files:
            console.print("[red]❌ No files generated. Check Ollama connection.[/red]")
            return

        project_dir = save_fullstack_project(project_name, files)
        state.add_history(task, "Full-Stack", state.model_id, True, str(project_dir))

        console.print()
        console.print(Rule("[bold bright_green]🎉 Full-Stack Project Complete![/bold bright_green]", style="green"))

    except Exception as exc:
        logger.exception("Fullstack error: %s", exc)
        console.print(f"[bold red]Error:[/bold red] {exc}")


def save_code(code: GeneratedCode) -> Path:
    """Save to language-specific subfolder under generated_code/v3/"""
    ext  = code.language.extension
    name = code.filename

    for known_ext in [".py",".js",".sh",".java",".cpp",".go",".rs",".html",
                      "_py","_js","_sh","_java","_cpp","_go","_rs","_html"]:
        while name.endswith(known_ext):
            name = name[:-len(known_ext)]
    name = name.rstrip("_-")
    name = re.sub(r'[^\w]', '_', name).strip('_') or "generated_script"

    if code.language.key == "java":
        m = re.search(r'public\s+class\s+(\w+)', code.code)
        if m:
            name = m.group(1)

    lang_dir = OUTPUT_DIR / code.language.key
    lang_dir.mkdir(parents=True, exist_ok=True)
    fp = lang_dir / (name + ext)
    fp.write_text(code.code, encoding="utf-8")
    code.filepath = fp
    logger.info("Saved: %s", fp)
    return fp


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════

def gen_system_prompt(lang: Language) -> str:
    extras = {
        "python":     "- if __name__ == '__main__': guard required\n- Use type hints\n- NEVER use input() — script must run fully automatically",
        "javascript": "- Use modern ES6+ syntax\n- Use console.log() for output",
        "bash":       "- Start with #!/bin/bash\n- Use echo for output",
        "java":       "- ONE public class, name must match filename\n- Include public static void main(String[] args)",
        "cpp":        "- Include all needed headers\n- Use int main()\n- Use cout for output",
        "go":         "- Use 'package main'\n- Import fmt\n- Use func main()",
        "rust":       "- Use fn main()\n- ONLY std library, NO extern crates\n- Use println! for output",
        "html":       "- Create a complete, beautiful HTML file with embedded CSS and JavaScript\n- Use modern CSS with gradients, animations, flexbox/grid\n- Make it visually stunning and responsive\n- All in one self-contained HTML file\n- No external dependencies — embed everything",
        "fullstack":  "IGNORED - handled separately",
    }.get(lang.key, "")

    return f"""\
You are an expert {lang.name} code generator.
Always respond using EXACTLY this structure:

##FILENAME##
<name_without_extension>
##DESCRIPTION##
<one sentence>
##DEPENDENCIES##
<pip packages for Python only, or: none>
##CODE##
<complete {lang.name} code — no markdown fences>
##END##

Rules:
- Start with ##FILENAME## immediately — no intro text
- No ``` fences inside ##CODE##
- Code must be complete and work as-is
{extras}
"""


def repair_system_prompt(lang: Language) -> str:
    return f"""\
You are an expert {lang.name} debugger. Fix the failing script.

##THOUGHT##
<what caused the error and exactly how to fix it>
##FILENAME##
<same filename>
##DESCRIPTION##
<one sentence>
##DEPENDENCIES##
<pip packages or: none>
##CODE##
<complete FIXED {lang.name} code — full file>
##END##

Rules:
- Start with ##THOUGHT## immediately
- Python ModuleNotFoundError → add package to ##DEPENDENCIES##
- CRITICAL: If TimeoutError → script uses input(), REMOVE all input() calls, use hardcoded demo values
- Write the COMPLETE fixed file
- No ``` fences
"""


# ═══════════════════════════════════════════════════════════════════
#  RICH UI
# ═══════════════════════════════════════════════════════════════════

PHASE_ICONS  = {Phase.THINKING:"🧠", Phase.ACTING:"✍️",
                 Phase.OBSERVING:"🔍", Phase.CORRECTING:"🔧",
                 Phase.SUCCEEDED:"🎉", Phase.FAILED:"💀"}
PHASE_COLORS = {Phase.THINKING:"bright_cyan", Phase.ACTING:"bright_magenta",
                 Phase.OBSERVING:"bright_yellow", Phase.CORRECTING:"bright_red",
                 Phase.SUCCEEDED:"bright_green", Phase.FAILED:"red"}
SYNTAX_MAP   = {"python":"python","javascript":"javascript","bash":"bash",
                "java":"java","cpp":"cpp","go":"go","rust":"rust","html":"html",
                "fullstack":"python"}


def banner(state: AgentState) -> None:
    lang_icons = " · ".join(f"{v.icon}{v.name}" for v in LANGUAGES.values())
    art = (
        "  ╔═══════════════════════════════════════════════════════╗\n"
        "  ║      AuraCode  ·  V3 PRO  ·  Self-Healing             ║\n"
        "  ║         [ Auto Language Detection ]                   ║\n"
        "  ║    Just describe what you want — I'll figure it out!  ║\n"
        "  ╚═══════════════════════════════════════════════════════╝"
    )
    console.print(Panel(
        Align(Text(art, style="bold bright_cyan"), align="center"),
        border_style="bright_blue", padding=(0, 1)))
    console.print(Panel(
        f"  [bold]Model  :[/bold] [cyan]{state.model_id}[/cyan]\n"
        f"  [bold]Output :[/bold] [dim]generated_code/v3/<language>/[/dim]\n\n"
        f"  [dim]{lang_icons}[/dim]",
        title="[bold]⚡ Configuration[/bold]",
        border_style="cyan", padding=(0, 2)))
    console.print(Panel(
        "  [bold cyan]Ctrl+L[/bold cyan]  Switch Model        "
        "[bold yellow]F2[/bold yellow]     View History\n"
        "  [bold red]Ctrl+C[/bold red]  Exit\n\n"
        "  [dim]No need to select language — just describe your task![/dim]",
        title="[bold]⌨  Hotkeys[/bold]",
        border_style="dim", padding=(0, 2)))
    console.print()


def phase_header(phase: Phase, attempt: int, max_r: int, lang: Language) -> None:
    c = PHASE_COLORS[phase]
    console.print()
    console.print(Rule(
        f"[bold {c}]{PHASE_ICONS[phase]}  {phase.name}  "
        f"—  Attempt {attempt}/{max_r}  ·  {lang.icon} {lang.name}[/bold {c}]",
        style=c))


def thought_panel(thought: str) -> None:
    if thought.strip():
        console.print(Panel(
            Text(thought[:800], style="italic bright_cyan"),
            title="[bold bright_cyan]🧠 Agent Reasoning[/bold bright_cyan]",
            border_style="bright_cyan", padding=(1, 2)))


def code_panel(code: GeneratedCode) -> None:
    name  = code.filepath.name if code.filepath else code.filename
    lexer = SYNTAX_MAP.get(code.language.key, "text")
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
    t.add_row("Status", f"[{color}]{'✅ SUCCESS' if r.success else '❌ FAILED'}[/{color}]")
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
        console.print(Panel("[dim]No tasks yet.[/dim]",
            title="[bold]📜 History[/bold]", border_style="dim"))
        return
    t = Table(box=box.ROUNDED, border_style="bright_blue")
    t.add_column("#",        style="dim",          width=3)
    t.add_column("Time",     style="dim",          width=19)
    t.add_column("Task",     style="bright_white", width=38)
    t.add_column("Language", style="cyan",         width=12)
    t.add_column("Status",   style="bold",         width=8)
    t.add_column("File",     style="dim",          width=22)
    for i, h in enumerate(history, 1):
        status = "[green]✅[/green]" if h["success"] else "[red]❌[/red]"
        task   = h["task"][:36] + "…" if len(h["task"]) > 36 else h["task"]
        fname  = Path(h["file"]).name if h["file"] else "—"
        t.add_row(str(i), h["timestamp"], task, h["language"], status, fname)
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


def success_panel(s: Session) -> None:
    elapsed = (datetime.datetime.now() - s.started_at).total_seconds()
    console.print()
    console.print(Rule("[bold bright_green]🎉  MISSION ACCOMPLISHED[/bold bright_green]", style="green"))
    console.print(Panel(
        f"[bold]Task:[/bold]      {s.task[:80]}\n"
        f"[bold]Language:[/bold]  {s.language.icon} {s.language.name}\n"
        f"[bold]Attempts:[/bold]  {s.total_attempts}\n"
        f"[bold]Elapsed:[/bold]   {elapsed:.1f}s\n"
        f"[bold]Model:[/bold]     {s.model_id}\n"
        f"[bold]Output:[/bold]    generated_code/v3/{s.language.key}/",
        title="[bold green]✅ Session Summary[/bold green]",
        border_style="green", padding=(1, 2)))


def failure_panel(s: Session, max_r: int) -> None:
    elapsed = (datetime.datetime.now() - s.started_at).total_seconds()
    console.print()
    console.print(Rule("[bold red]💀  MAX RETRIES EXCEEDED[/bold red]", style="red"))
    console.print(Panel(
        f"[bold]Task:[/bold]      {s.task[:80]}\n"
        f"[bold]Language:[/bold]  {s.language.icon} {s.language.name}\n"
        f"[bold]Attempts:[/bold]  {s.total_attempts}/{max_r}\n"
        f"[bold]Elapsed:[/bold]   {elapsed:.1f}s",
        title="[bold red]❌ Session Summary[/bold red]",
        border_style="red", padding=(1, 2)))


# ═══════════════════════════════════════════════════════════════════
#  SELF-HEALING AGENT
# ═══════════════════════════════════════════════════════════════════

class AuraCodeAgentV3Pro:
    """
    V3 PRO: Auto-detects language, supports HTML/websites,
    self-healing ReAct loop.
    """

    def __init__(self, state: AgentState, max_retries: int = MAX_RETRIES):
        self.state       = state
        self.max_retries = max_retries
        self._env        = Env()

    def _llm(self, sys_prompt: str, messages: list[dict], spinner: str) -> str:
        with Progress(
            SpinnerColumn(spinner_name="dots2", style="bright_cyan"),
            TextColumn(f"[bright_cyan]  {spinner}[/bright_cyan]"),
            TimeElapsedColumn(), transient=True, console=console,
        ) as p:
            p.add_task("l", total=None)
            resp = self.state.get_client().chat.completions.create(
                model=self.state.model_id,
                messages=[{"role": "system", "content": sys_prompt}] + messages,
                temperature=0.1,
            )
        return resp.choices[0].message.content

    def _initial(self, task: str, lang: Language) -> GeneratedCode:
        raw = self._llm(
            gen_system_prompt(lang),
            [{"role": "user",
              "content": f"Create a {lang.name} script/file for:\n\n{task}"}],
            f"Generating {lang.icon} {lang.name} code…",
        )
        if not raw.strip():
            raise ValueError("Model returned empty response.")
        return GeneratedCode.from_parsed(parse_llm_response(raw, 1), lang, 1)

    def _repair(self, task: str, broken: GeneratedCode, result: ExecResult,
                attempt: int, history: list[dict]) -> GeneratedCode:
        lang = broken.language
        msg  = (
            f"TASK: {task}\nERROR TYPE: {result.error_type}\n\n"
            f"STDERR:\n{result.stderr[:2000]}\n\n"
            f"STDOUT:\n{result.stdout[:300]}\n\n"
            f"BROKEN CODE:\n{broken.code}\n\nFix it completely."
        )
        raw = self._llm(
            repair_system_prompt(lang),
            history + [{"role": "user", "content": msg}],
            f"Self-healing · attempt #{attempt}…",
        )
        if not raw.strip():
            raise ValueError("Empty repair response.")
        return GeneratedCode.from_parsed(
            parse_llm_response(raw, attempt), lang, attempt)

    def run(self, task: str, lang: Language) -> Session:
        session = Session(task=task, language=lang, model_id=self.state.model_id)
        history: list[dict] = []
        current: Optional[GeneratedCode] = None
        result:  Optional[ExecResult]    = None

        for attempt in range(1, self.max_retries + 1):

            # ── THINK ────────────────────────────────────────────
            phase_header(Phase.THINKING, attempt, self.max_retries, lang)
            try:
                if attempt == 1:
                    current = self._initial(task, lang)
                else:
                    current = self._repair(task, current, result, attempt, history)
                    thought_panel(current.thought)
            except ValueError as exc:
                console.print(Panel(str(exc)[:400],
                    title="[red]⚠ Parse Error — retrying[/red]",
                    border_style="red"))
                if current is None:
                    session.final_phase = Phase.FAILED
                    return session
                history.append({"role": "user",
                    "content": "Response could not be parsed. Follow output format strictly."})
                continue

            history.append({"role": "assistant",
                "content": f"##THOUGHT##\n{current.thought}\n"
                           f"##FILENAME##\n{current.filename}\n"
                           f"##CODE##\n{current.code}\n##END##"})

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
                TextColumn(f"[yellow]  Running {lang.icon} {lang.name}…[/yellow]"),
                TimeElapsedColumn(), transient=True, console=console,
            ) as p:
                p.add_task("r", total=None)
                result = self._env.run(fp, lang)

            exec_panel(result, attempt)
            session.log(attempt, current, result)

            if result.success:
                session.final_phase = Phase.SUCCEEDED
                success_panel(session)
                self._save_session(session)
                self.state.add_history(
                    task, lang.name, self.state.model_id, True, str(fp))
                return session

            if attempt < self.max_retries:
                phase_header(Phase.CORRECTING, attempt, self.max_retries, lang)
                console.print(Panel(
                    f"[yellow]Error:[/yellow] [bold red]{result.error_type}[/bold red]\n"
                    "[yellow]Feeding traceback to LLM for repair…[/yellow]",
                    border_style="yellow"))
                history.append({"role": "user",
                    "content": f"Attempt #{attempt} failed.\n"
                               f"Error: {result.error_type}\n"
                               f"STDERR:\n{result.stderr[:1500]}"})

        session.final_phase = Phase.FAILED
        failure_panel(session, self.max_retries)
        self._save_session(session)
        self.state.add_history(
            task, lang.name, self.state.model_id, False,
            str(current.filepath) if current and current.filepath else "")
        return session

    def _save_session(self, s: Session) -> None:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = LOG_DIR / f"v3pro_{ts}.json"
        path.write_text(s.to_json(), encoding="utf-8")
        console.print(f"  [dim]📝 Log → {path}[/dim]")


# ═══════════════════════════════════════════════════════════════════
#  INTERACTIVE CLI
# ═══════════════════════════════════════════════════════════════════

def build_keybindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-l")
    def _(event):
        event.app.exit(result="__SWITCH_MODEL__")

    @kb.add("f2")
    def _(event):
        event.app.exit(result="__SHOW_HISTORY__")

    return kb


def run_cli(state: AgentState) -> None:
    agent   = AuraCodeAgentV3Pro(state)
    kb      = build_keybindings()
    session = PromptSession(
        history=InMemoryHistory(),
        key_bindings=kb,
        style=Style.from_dict({"bottom-toolbar": "bg:#1a1a2e fg:#888888"}),
    )

    banner(state)

    while True:
        try:
            toolbar = HTML(
                f" <b>Model:</b> {state.model_id}  "
                f"<b>History:</b> {len(state.history)} tasks  "
                "│  <b>Ctrl+L</b> Model  <b>F2</b> History  "
                "│  <i>Auto-detects language from your prompt</i>"
            )
            task = session.prompt(
                HTML("<ansicyan><b>► </b></ansicyan>"
                     "<ansigreen>Describe what you want to build</ansigreen>"
                     "<ansicyan><b>: </b></ansicyan>"),
                bottom_toolbar=toolbar,
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        # ── Hotkeys ───────────────────────────────────────────────
        if task == "__SWITCH_MODEL__":
            models = get_ollama_models()
            if not models:
                console.print("[red]Could not fetch models. Is Ollama running?[/red]")
                continue
            state.available_models = models
            show_model_selector(models, state.model_id)
            console.print()
            try:
                choice = console.input(
                    f"[bold cyan]Enter number (1-{len(models)}) or model name: [/bold cyan]"
                ).strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        state.set_model(models[idx])
                        console.print(f"[green]✓ Model → [bold]{state.model_id}[/bold][/green]")
                elif choice:
                    state.set_model(choice)
                    console.print(f"[green]✓ Model → [bold]{state.model_id}[/bold][/green]")
            except KeyboardInterrupt:
                pass
            continue

        if task == "__SHOW_HISTORY__":
            show_history(state.history)
            continue

        if not task or not task.strip():
            continue
        if task.strip().lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        # ── Auto-detect language ──────────────────────────────────
        task = task.strip()
        lang_key = detect_language(task)

        if lang_key is None:
            # Ambiguous — ask user
            console.print(Panel(
                f"[yellow]Hmm, aku tidak yakin bahasa apa yang kamu maksud untuk task ini:[/yellow]\n"
                f"[bold]'{task[:60]}'[/bold]",
                border_style="yellow"))
            lang = ask_language_choice()
        else:
            lang = LANGUAGES[lang_key]
            console.print(Panel(
                f"[green]🎯 Auto-detected:[/green] {lang.icon} [bold]{lang.name}[/bold]\n"
                f"[dim]Based on your prompt keywords[/dim]",
                border_style="green"))

        # ── Fullstack: use dedicated generator ────────────────────
        if lang.key == "fullstack":
            try:
                run_fullstack(state, task)
            except Exception as exc:
                logger.exception("Unexpected: %s", exc)
                console.print(f"[bold red]Error:[/bold red] {exc}")
            continue

        try:
            agent.run(task, lang)
        except Exception as exc:
            logger.exception("Unexpected: %s", exc)
            console.print(f"[bold red]Error:[/bold red] {exc}")
            if "connection" in str(exc).lower():
                console.print("[yellow]→ Make sure Ollama is running[/yellow]")


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