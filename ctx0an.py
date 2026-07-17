#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctx0an.py — Ctx0an: an autonomous software-development agent for Termux.

Ctx0an is a single-file, dependency-light CLI agent. It uses Google's Gemini
models through the modern ``google-genai`` SDK and can *act* on your device:
reading, writing and patching files, and executing shell commands — every
command gated behind an explicit ``[Y/n]`` keyboard confirmation.

Setup on Termux
---------------
    pkg install python
    pip install google-genai rich
    export GEMINI_API_KEY="your-api-key"        # add to ~/.bashrc to persist

    # optional: make it a real command
    chmod +x ctx0an.py && cp ctx0an.py $PREFIX/bin/ctx0an

Usage
-----
    ctx0an                                       # interactive workspace mode
    ctx0an "create a flask app and run it on port 8080"
    ctx0an -m flash "fix the traceback in main.py and rerun it"

Modes
-----
* Single-task mode: pass a task as an argument; the agent works until the
  task is complete, then exits.
* Interactive workspace mode: run with no arguments to open a session where
  conversation history is preserved between turns.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import http.server
import json
import os
import queue
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

__version__ = "1.0.0"

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-pro"          # advanced coding model (default)
FALLBACK_MODEL = "gemini-2.5-flash"       # cheaper/faster fallback
MODEL_ALIASES = {
    "pro": "gemini-2.5-pro",
    "flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "haiku": "claude-3-5-haiku-20241022",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "4o": "gpt-4o",
    "4o-mini": "gpt-4o-mini"
}

MODEL_PROVIDERS = {
    "gemini-2.5-pro": "gemini",
    "gemini-2.5-flash": "gemini",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai"
}

MODEL_DESCRIPTIONS = {
    "gemini-2.5-pro": "Google Gemini 2.5 Pro",
    "gemini-2.5-flash": "Google Gemini 2.5 Flash",
    "claude-3-5-sonnet-20241022": "Anthropic Claude 3.5 Sonnet",
    "claude-3-5-haiku-20241022": "Anthropic Claude 3.5 Haiku",
    "gpt-4o": "OpenAI GPT-4o",
    "gpt-4o-mini": "OpenAI GPT-4o Mini"
}

MAX_LINES = 5_000          # hard line cap for any text sent to the model
MAX_CHARS = 60_000         # hard character cap for any text sent to the model
MAX_FILE_BYTES = 2_000_000 # refuse to read files larger than 2 MB
COMMAND_TIMEOUT = 120      # seconds before a shell command is killed
MAX_AGENT_STEPS = 40       # safety cap on consecutive tool-calling turns
API_MAX_RETRIES = 3        # retries on transient API failures

GUI_MODE = False
gui_event_queue = queue.Queue()
gui_confirm_event = threading.Event()
gui_confirm_response = False
global_agent = None


# --------------------------------------------------------------------------
# Optional dependency: rich (attempt import -> attempt install -> ANSI fallback)
# --------------------------------------------------------------------------

def _import_rich() -> bool:
    """Return True if `rich` is importable, installing it first if needed."""
    if os.environ.get("CTX0AN_NO_RICH"):          # escape hatch / testing
        return False
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        pass
    print("[ctx0an] 'rich' not found — attempting: pip install rich",
          file=sys.stderr)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "rich"],
            check=True, timeout=300,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        import rich  # noqa: F401
        return True
    except Exception:
        print("[ctx0an] could not install 'rich' — using plain ANSI output.",
              file=sys.stderr)
        return False


HAS_RICH = _import_rich()
if HAS_RICH:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Confirm
    from rich.syntax import Syntax
    from rich.text import Text


# --------------------------------------------------------------------------
# Plain-ANSI helpers (used only when rich is unavailable)
# --------------------------------------------------------------------------

class _ANSI:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


def _ansi(text: str, *styles: str) -> str:
    """Wrap text in ANSI styles, but only on a real terminal."""
    if not sys.stdout.isatty():
        return text
    return "".join(styles) + text + _ANSI.RESET


class _PlainSpinner:
    """Minimal braille spinner for the no-rich fallback path."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_PlainSpinner":
        if not sys.stdout.isatty():
            print(f"... {self.message}")
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{self.FRAMES[i % len(self.FRAMES)]} "
                             f"{self.message}   ")
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)

    def __exit__(self, *exc: object) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
        if sys.stdout.isatty():
            sys.stdout.write("\r" + " " * (len(self.message) + 8) + "\r")
            sys.stdout.flush()
        return False


def _shortify(value: Any, limit: int = 100) -> str:
    """One-line, length-capped preview of a tool argument value."""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            s = str(value)
    s = s.replace("\n", "\\n")
    return s if len(s) <= limit else s[:limit] + "..."


# --------------------------------------------------------------------------
# UI layer: every rendering decision lives here so the agent logic stays clean
# --------------------------------------------------------------------------

class UI:
    """Terminal UI with a rich implementation and a plain-ANSI fallback."""

    def __init__(self) -> None:
        self.console: Console | None = Console() if HAS_RICH else None

    # -- generic messages --------------------------------------------------

    def echo(self, text: str = "") -> None:
        if self.console:
            self.console.print(text, markup=False)
        else:
            print(text)

    def info(self, msg: str) -> None:
        if self.console:
            self.console.print(msg, style="dim", markup=False)
        else:
            print(_ansi(msg, _ANSI.DIM))

    def success(self, msg: str) -> None:
        if self.console:
            self.console.print(msg, style="green", markup=False)
        else:
            print(_ansi(msg, _ANSI.GREEN))

    def warn(self, msg: str) -> None:
        if self.console:
            self.console.print(f"Warning: {msg}", style="yellow", markup=False)
        else:
            print(_ansi(f"Warning: {msg}", _ANSI.YELLOW), file=sys.stderr)

    def error(self, msg: str) -> None:
        if self.console:
            self.console.print(f"Error: {msg}", style="bold red", markup=False)
        else:
            print(_ansi(f"Error: {msg}", _ANSI.RED, _ANSI.BOLD),
                  file=sys.stderr)

    # -- structured rendering ----------------------------------------------

    def banner(self, model: str, mode: str) -> None:
        """Startup banner showing model, working directory and safety note."""
        if self.console:
            body = Text()
            body.append("Ctx0an", style="bold cyan")
            body.append(f"  v{__version__} · {mode} mode\n", style="dim")
            body.append(f"model: {model}\n")
            body.append(f"cwd:   {os.getcwd()}\n", style="dim")
            body.append(
                "shell commands always require your [Y/n] confirmation",
                style="dim italic",
            )
            if mode == "interactive":
                body.append("\nType /help for session commands, "
                            "exit to quit.", style="dim")
            self.console.print(Panel(body, border_style="cyan", expand=False))
        else:
            print(_ansi(f"=== Ctx0an v{__version__} ({mode} mode) ===",
                        _ANSI.CYAN, _ANSI.BOLD))
            print(f"model: {model}")
            print(f"cwd:   {os.getcwd()}")
            print("shell commands require your [Y/n] confirmation")
            if mode == "interactive":
                print("type /help for session commands, exit to quit")

    def assistant(self, markdown_text: str) -> None:
        """Render a model reply (markdown when rich is available)."""
        if GUI_MODE:
            gui_event_queue.put({"type": "text", "content": markdown_text})
        if self.console:
            try:
                self.console.print(Markdown(markdown_text))
                return
            except Exception:
                pass                                # fall through to plain
        print(markdown_text)

    def code(self, code_text: str, lang: str = "") -> None:
        """Syntax-highlighted code block (or indented plain text)."""
        if self.console:
            self.console.print(
                Syntax(code_text, lang or "text", theme="monokai",
                       word_wrap=True)
            )
        else:
            for line in code_text.splitlines():
                print(_ansi("  | ", _ANSI.DIM) + line)

    def diff(self, old: str, new: str, path: str) -> None:
        """Show a unified diff for a successful patch."""
        diff_lines = difflib.unified_diff(
            old.splitlines(), new.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
        )
        text = "\n".join(diff_lines)
        if text.strip():
            self.code(text, "diff")

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        """Spinner shown while the model is thinking."""
        if self.console:
            with self.console.status(f"[bold cyan]{message}[/]",
                                     spinner="dots"):
                yield
        else:
            with _PlainSpinner(message):
                yield

    # -- tool-call visualisation -------------------------------------------

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        summary = ", ".join(f"{k}={_shortify(v)}" for k, v in args.items())
        if GUI_MODE:
            gui_event_queue.put({"type": "tool_call", "name": name, "args": args})
        if self.console:
            self.console.print(
                f"[bold yellow]▸ tool[/] [bold]{name}[/] [dim]{summary}[/]"
            )
        else:
            print(_ansi(f"> tool {name} {summary}",
                        _ANSI.YELLOW, _ANSI.BOLD))

    def tool_result(self, text: str, max_lines: int = 12) -> None:
        """Dim preview of a tool result (the model receives the full text)."""
        if GUI_MODE:
            gui_event_queue.put({"type": "tool_result", "content": text})
        lines = str(text).splitlines()
        extra = len(lines) - max_lines
        body = "\n".join(lines[:max_lines])
        if extra > 0:
            body += f"\n... ({extra} more lines)"
        if self.console:
            self.console.print(
                Panel(Text(body), border_style="bright_black", expand=False)
            )
        else:
            for line in body.splitlines():
                print(_ansi("  " + line, _ANSI.DIM))

    # -- user interaction ----------------------------------------------------

    def confirm(self, question: str) -> bool:
        """[Y/n] confirmation prompt. Defaults to Yes on Enter; any
        interrupt/EOF means No (fail closed)."""
        if GUI_MODE:
            gui_event_queue.put({"type": "confirm_request", "question": question})
            gui_confirm_event.clear()
            success = gui_confirm_event.wait(timeout=300)
            if not success:
                gui_event_queue.put({"type": "error", "content": "Confirmation timed out."})
                return False
            return gui_confirm_response

        if self.console:
            try:
                return bool(Confirm.ask(f"[bold cyan]{question}[/]",
                                        default=True))
            except (EOFError, KeyboardInterrupt):
                self.console.print("")
                return False
        try:
            answer = input(f"{question} [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("")
            return False
        return answer in ("", "y", "yes")

    def user_prompt(self) -> str:
        if self.console:
            return self.console.input("[bold green]you[/] [dim]>[/] ")
        return input("you > ")


# Single shared UI instance used by the tools and the agent.
ui = UI()


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _resolve_path(path: str) -> Path:
    """Expand ~ and $VARS, then resolve to an absolute path."""
    return Path(os.path.expandvars(os.path.expanduser(path.strip()))).resolve()


def _fmt_size(num_bytes: float) -> str:
    """Human-readable file size, e.g. '512B', '3.2KB', '1.5MB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)}B"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _truncate(text: str, source: str = "output") -> str:
    """Cap text at MAX_LINES / MAX_CHARS, keeping the head AND the tail so
    that tracebacks (which live at the end) survive truncation."""
    original_len = len(text)
    lines = text.split("\n")
    total_lines = len(lines)
    if total_lines > MAX_LINES:
        half = MAX_LINES // 2
        lines = (
            lines[:half]
            + [f"... [{source} truncated: {total_lines} lines total, "
               f"showing first {half} and last {half}] ..."]
            + lines[total_lines - half:]
        )
    out = "\n".join(lines)
    if len(out) > MAX_CHARS:
        keep = MAX_CHARS // 2
        out = (
            out[:keep]
            + f"\n... [{source} truncated: middle removed, "
              f"{original_len} chars total] ...\n"
            + out[-keep:]
        )
    return out


def _detect_shell() -> str | None:
    """Locate a usable shell. On Termux /bin/sh does not exist, so we must
    resolve sh/bash from PATH ($PREFIX/bin is on PATH there)."""
    for candidate in ("bash", "sh"):
        found = shutil.which(candidate)
        if found:
            return found
    return None                     # subprocess falls back to /bin/sh


SHELL_BIN = _detect_shell()

# Heuristic patterns for potentially destructive commands. They never block
# execution by themselves — they only upgrade the confirmation warning.
_DANGEROUS_PATTERNS = (
    r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+/(\s|$)",   # rm -rf /
    r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+~",          # rm -rf ~
    r"\bmkfs[.\w]*\b",                              # mkfs.*
    r"\bdd\b[^\n]*\bof=/dev/",                      # dd of=/dev/...
    r">\s*/dev/(sd|mmc|nvme)",                      # overwrite block device
    r":\(\)\s*\{",                                  # fork bomb
    r"\bshutdown\b|\breboot\b",
)
_DANGEROUS_RE = re.compile("|".join(_DANGEROUS_PATTERNS))


# --------------------------------------------------------------------------
# Agent tools — the five functions the model can invoke via Function Calling
# --------------------------------------------------------------------------

def run_command(command: str) -> str:
    """Execute a shell command in Termux and return its output.

    SAFETY: the command is shown to the user and only runs after an explicit
    [Y/n] keyboard confirmation. If there is no interactive terminal, the
    command is refused. Times out after COMMAND_TIMEOUT seconds.
    """
    ui.code(command, "bash")
    if _DANGEROUS_RE.search(command):
        ui.warn("this command looks potentially DESTRUCTIVE — "
                "review it carefully before approving.")

    if not sys.stdin.isatty():
        msg = ("Command NOT executed: no interactive terminal is available "
               "for user confirmation.")
        ui.warn(msg)
        return msg

    if not ui.confirm("Execute this command?"):
        msg = ("Command declined by the user. Do not retry the same "
               "command; choose a different approach or explain the "
               "situation and ask how to proceed.")
        ui.warn(msg)
        return msg

    ui.info(f"running (timeout {COMMAND_TIMEOUT}s) ...")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            executable=SHELL_BIN,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout if isinstance(exc.stdout, str) else ""
        result = _truncate(
            f"$ {command}\n"
            f"[TIMEOUT: killed after {COMMAND_TIMEOUT}s]\n"
            f"--- partial output ---\n{partial}",
            "command output",
        )
        ui.tool_result(result)
        return result
    except Exception as exc:  # e.g. OSError spawning the shell
        msg = f"Error: failed to execute command: {type(exc).__name__}: {exc}"
        ui.error(msg)
        return msg

    parts = [f"$ {command}", f"[exit code {proc.returncode}]"]
    if proc.stdout.strip():
        parts.append(f"--- stdout ---\n{proc.stdout.rstrip()}")
    if proc.stderr.strip():
        parts.append(f"--- stderr ---\n{proc.stderr.rstrip()}")
    if not proc.stdout.strip() and not proc.stderr.strip():
        parts.append("(no output)")
    result = _truncate("\n".join(parts), "command output")
    ui.tool_result(result)
    return result


def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its content (truncated if huge)."""
    try:
        p = _resolve_path(path)
    except Exception as exc:
        return f"Error: invalid path '{path}': {exc}"
    if not p.exists():
        return f"Error: file not found: {path}"
    if p.is_dir():
        return f"Error: '{path}' is a directory — use list_directory instead."
    try:
        raw = p.read_bytes()
    except PermissionError:
        return f"Error: permission denied reading: {path}"
    except OSError as exc:
        return f"Error: cannot read '{path}': {exc}"
    if len(raw) > MAX_FILE_BYTES:
        return (f"Error: file too large ({len(raw)} bytes > "
                f"{MAX_FILE_BYTES}). Inspect it with run_command instead, "
                f"e.g. 'head -n 200 {path}' or 'grep'.")
    if b"\x00" in raw[:8192]:
        return f"Error: '{path}' appears to be a binary file; not dumping it."
    text = raw.decode("utf-8", errors="replace")
    ui.tool_result(f"read {p} — {text.count(chr(10)) + 1} lines, "
                   f"{len(raw)} bytes")
    return _truncate(text, f"file {path}")


def write_file(path: str, content: str) -> str:
    """Write a new file or completely overwrite an existing one (UTF-8).
    Parent directories are created automatically."""
    try:
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except PermissionError:
        return f"Error: permission denied writing: {path}"
    except OSError as exc:
        return f"Error: cannot write '{path}': {exc}"
    msg = (f"OK: wrote {p} ({content.count(chr(10)) + 1} lines, "
           f"{len(content.encode('utf-8'))} bytes)")
    ui.tool_result(msg)
    return msg


def edit_file_patch(path: str, search_block: str, replace_block: str) -> str:
    """Replace exactly one occurrence of search_block with replace_block.

    This function is resilient: it normalizes line endings and, if an exact
    match fails, performs fuzzy matching and outputs diagnostic diffs to
    help self-healing if a patch mismatch occurs.
    """
    try:
        p = _resolve_path(path)
    except Exception as exc:
        return f"Error: invalid path '{path}': {exc}"
    if not p.exists():
        return f"Error: file not found: {path}"
    if search_block == "":
        return "Error: search_block must not be empty."
    try:
        original = p.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Error: permission denied reading: {path}"
    except OSError as exc:
        return f"Error: cannot read '{path}': {exc}"

    count = original.count(search_block)
    if count == 1:
        patched = original.replace(search_block, replace_block, 1)
        try:
            p.write_text(patched, encoding="utf-8")
        except PermissionError:
            return f"Error: permission denied writing: {path}"
        except OSError as exc:
            return f"Error: cannot write '{path}': {exc}"
        ui.diff(search_block, replace_block, str(p))
        msg = (f"OK: patched {p} (1 block: {search_block.count(chr(10)) + 1} -> "
               f"{replace_block.count(chr(10)) + 1} lines)")
        ui.tool_result(msg)
        return msg

    if count > 1:
        return (f"Error: search_block matches {count} locations in {path}. "
                "Include more surrounding context so it matches exactly once.")

    # count == 0: try resilient / fuzzy matching
    # 1. Try line endings normalization (CRLF vs LF)
    orig_norm = original.replace("\r\n", "\n")
    search_norm = search_block.replace("\r\n", "\n")
    if orig_norm.count(search_norm) == 1:
        has_crlf = "\r\n" in original
        replace_norm = replace_block.replace("\r\n", "\n")
        patched_norm = orig_norm.replace(search_norm, replace_norm, 1)
        patched = patched_norm.replace("\n", "\r\n") if has_crlf else patched_norm
        try:
            p.write_text(patched, encoding="utf-8")
        except PermissionError:
            return f"Error: permission denied writing: {path}"
        except OSError as exc:
            return f"Error: cannot write '{path}': {exc}"
        ui.diff(search_block, replace_block, str(p))
        msg = f"OK: patched {p} after line ending normalization."
        ui.tool_result(msg)
        return msg

    # 2. Sliding window fuzzy matching
    orig_lines = original.splitlines()
    search_lines = search_block.splitlines()
    n_search = len(search_lines)
    
    if n_search > 0 and len(orig_lines) >= n_search:
        best_ratio = 0.0
        best_idx = -1
        
        # Sliding window over the file lines
        for i in range(len(orig_lines) - n_search + 1):
            window = orig_lines[i:i + n_search]
            ratio = difflib.SequenceMatcher(None, "\n".join(window), "\n".join(search_lines)).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
                if ratio == 1.0:
                    break

        # Auto-patch if extremely similar (e.g. >= 96% match, which usually implies tiny whitespace or single char difference)
        if best_ratio >= 0.96 and best_idx != -1:
            new_lines = orig_lines[:best_idx] + replace_block.splitlines() + orig_lines[best_idx + n_search:]
            line_ending = "\r\n" if "\r\n" in original else "\n"
            patched = line_ending.join(new_lines)
            try:
                p.write_text(patched, encoding="utf-8")
            except Exception as exc:
                return f"Error: cannot write fuzzy patch: {exc}"
            
            orig_match = "\n".join(orig_lines[best_idx:best_idx + n_search])
            ui.diff(orig_match, replace_block, str(p))
            msg = f"OK: fuzzy patched {p} at lines {best_idx+1}-{best_idx+n_search} (similarity {best_ratio:.2%})."
            ui.tool_result(msg)
            return msg

        # Generate detailed diagnostics diff for self-healing if similarity is >= 70%
        if best_ratio >= 0.70 and best_idx != -1:
            matched_window = orig_lines[best_idx:best_idx + n_search]
            diff = difflib.unified_diff(
                search_lines,
                matched_window,
                fromfile="your_search_block",
                tofile=f"actual_file_lines_{best_idx+1}_to_{best_idx+n_search}",
                lineterm=""
            )
            diff_str = "\n".join(list(diff))
            return (
                f"Error: search_block not found in {path}, but found a close match ({best_ratio:.1%} similarity) "
                f"at lines {best_idx+1} to {best_idx+n_search}.\n"
                f"--- DIFF BETWEEN YOUR SEARCH BLOCK AND FILE CONTENT ---\n"
                f"{diff_str}\n"
                f"------------------------------------------------------\n"
                f"Please update your search_block to match the file exactly (including indentation and spaces)."
            )

    hint = ""
    if search_block.replace("\r\n", "\n") in original.replace("\r\n", "\n"):
        hint = " Note: the file seems to use different line endings."
    return (f"Error: search_block not found in {path}.{hint} "
            f"Call read_file('{path}') first and copy the exact block, "
            "indentation included.")


def list_directory(path: str = ".") -> list:
    """List the contents of a directory: subdirectories first (trailing /),
    then files with sizes. Capped at 200 entries."""
    try:
        p = _resolve_path(path)
    except Exception as exc:
        return [f"Error: invalid path '{path}': {exc}"]
    if not p.exists():
        return [f"Error: no such path: {path}"]
    if not p.is_dir():
        return [f"Error: not a directory: {path}"]
    try:
        items = sorted(
            p.iterdir(),
            key=lambda x: (not x.is_dir(), x.name.lower()),
        )
    except PermissionError:
        return [f"Error: permission denied: {path}"]
    except OSError as exc:
        return [f"Error: cannot list '{path}': {exc}"]

    entries: list[str] = []
    for item in items[:200]:
        try:
            if item.is_dir():
                entries.append(item.name + "/")
            else:
                entries.append(
                    f"{item.name} ({_fmt_size(item.stat().st_size)})"
                )
        except OSError:
            entries.append(item.name)
    if len(items) > 200:
        entries.append(f"... and {len(items) - 200} more entries")
    ui.tool_result(f"{path}: {len(items)} entries\n" + "\n".join(entries))
    return entries


def get_file_outline(path: str) -> str:
    """Extract a structural outline (classes, methods, functions) from a Python source file using AST.
    
    Returns a clean hierarchy with line numbers and signatures, helping to understand large files without reading them fully.
    """
    try:
        p = _resolve_path(path)
    except Exception as exc:
        return f"Error: invalid path '{path}': {exc}"
    if not p.exists():
        return f"Error: file not found: {path}"
    if p.is_dir():
        return f"Error: '{path}' is a directory."
    
    if p.suffix != ".py":
        return f"Error: outline extraction is only supported for Python (.py) files. Got: {p.suffix}"
        
    try:
        code = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"
        
    try:
        tree = ast.parse(code, filename=str(p))
    except SyntaxError as exc:
        return f"Error parsing Python file (SyntaxError): {exc}"
        
    outline_lines = []
    
    def _ast_unparse(node) -> str:
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Attribute):
            return f"{_ast_unparse(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return f"{_ast_unparse(node.value)}[{_ast_unparse(node.slice)}]"
        if hasattr(ast, "Index") and isinstance(node, ast.Index):
            return _ast_unparse(node.value)
        return ""
    
    class OutlineVisitor(ast.NodeVisitor):
        def __init__(self):
            self.indent = 0
            
        def visit_ClassDef(self, node: ast.ClassDef):
            decorators = [f"@{_ast_unparse(d)}" for d in node.decorator_list]
            dec_prefix = "  " * self.indent
            for dec in decorators:
                if dec.strip() and dec != "@":
                    outline_lines.append(f"{dec_prefix}{dec}")
            
            bases = [_ast_unparse(b) for b in node.bases]
            bases_str = f"({', '.join(bases)})" if bases else ""
            outline_lines.append(f"{dec_prefix}class {node.name}{bases_str}:  # Line {node.lineno}")
            
            doc = ast.get_docstring(node)
            if doc:
                first_line = doc.splitlines()[0].strip()
                outline_lines.append(f"{dec_prefix}    \"\"\"{first_line}\"\"\"")
                
            self.indent += 1
            self.generic_visit(node)
            self.indent -= 1
            
        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._visit_func(node)
            
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._visit_func(node, is_async=True)
            
        def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool = False):
            args_list = []
            if hasattr(node.args, "posonlyargs") and node.args.posonlyargs:
                for arg in node.args.posonlyargs:
                    annotation = f": {_ast_unparse(arg.annotation)}" if arg.annotation else ""
                    args_list.append(f"{arg.arg}{annotation}")
                args_list.append("/")
            for arg in node.args.args:
                annotation = f": {_ast_unparse(arg.annotation)}" if arg.annotation else ""
                args_list.append(f"{arg.arg}{annotation}")
            if node.args.vararg:
                annotation = f": {_ast_unparse(node.args.vararg.annotation)}" if node.args.vararg.annotation else ""
                args_list.append(f"*{node.args.vararg.arg}{annotation}")
            if node.args.kwonlyargs:
                if not node.args.vararg:
                    args_list.append("*")
                for arg in node.args.kwonlyargs:
                    annotation = f": {_ast_unparse(arg.annotation)}" if arg.annotation else ""
                    args_list.append(f"{arg.arg}{annotation}")
            if node.args.kwarg:
                annotation = f": {_ast_unparse(node.args.kwarg.annotation)}" if node.args.kwarg.annotation else ""
                args_list.append(f"**{node.args.kwarg.arg}{annotation}")
                
            args_str = ", ".join(args_list)
            ret_annotation = f" -> {_ast_unparse(node.returns)}" if node.returns else ""
            prefix = "async " if is_async else ""
            
            dec_prefix = "  " * self.indent
            decorators = [f"@{_ast_unparse(d)}" for d in node.decorator_list]
            for dec in decorators:
                if dec.strip() and dec != "@":
                    outline_lines.append(f"{dec_prefix}{dec}")
                
            outline_lines.append(f"{dec_prefix}{prefix}def {node.name}({args_str}){ret_annotation}:  # Line {node.lineno}")
            
            doc = ast.get_docstring(node)
            if doc:
                first_line = doc.splitlines()[0].strip()
                outline_lines.append(f"{dec_prefix}    \"\"\"{first_line}\"\"\"")
            
    visitor = OutlineVisitor()
    visitor.visit(tree)
    
    if not outline_lines:
        return f"OK: No classes or functions found in {path}"
        
    result = f"Structure outline of {p}:\n" + "\n".join(outline_lines)
    ui.tool_result(result)
    return result


def search_grep(query: str, path: str = ".", extension_filter: str = "") -> str:
    """Search for a text pattern or regex query recursively across files.
    
    Filters binary files and standard ignored folders (.git, node_modules, etc.).
    """
    try:
        start_dir = _resolve_path(path)
    except Exception as exc:
        return f"Error: invalid path '{path}': {exc}"
    if not start_dir.exists():
        return f"Error: path does not exist: {path}"
    if not start_dir.is_dir():
        return f"Error: path is not a directory: {path}"
        
    ignored_dirs = {
        ".git", ".github", ".venv", "venv", "node_modules", "__pycache__",
        "dist", "build", ".sdk", ".gradle", "bin", "obj"
    }
    
    is_regex = False
    try:
        if any(c in query for c in "*+?^$()[]{}|\\"):
            rx = re.compile(query, re.IGNORECASE)
            is_regex = True
        else:
            rx = None
    except Exception:
        rx = None
        
    results = []
    match_count = 0
    max_matches = 100
    
    exts = []
    if extension_filter:
        exts = [e.strip().lower() for e in extension_filter.split(",")]
        exts = [e if e.startswith(".") else f".{e}" for e in exts]
        
    for root, dirs, files in os.walk(start_dir):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            file_path = Path(root) / file
            
            if exts and not any(file_path.suffix.lower() == e for e in exts):
                continue
                
            try:
                if file_path.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue
                
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, 1):
                        matched = False
                        if is_regex and rx:
                            if rx.search(line):
                                matched = True
                        else:
                            if query.lower() in line.lower():
                                matched = True
                                
                        if matched:
                            rel_path = os.path.relpath(file_path, start_dir)
                            results.append(f"{rel_path}:{line_no}: {line.strip()}")
                            match_count += 1
                            if match_count >= max_matches:
                                break
            except Exception:
                continue
                
            if match_count >= max_matches:
                break
        if match_count >= max_matches:
            break
            
    if not results:
        return f"OK: No matches found for '{query}'"
        
    header = f"Found {match_count} matches for '{query}':\n"
    if match_count >= max_matches:
        header = f"Found {match_count}+ matches for '{query}' (capped at {max_matches}):\n"
        
    result_str = header + "\n".join(results)
    ui.tool_result(result_str, max_lines=15)
    return result_str


# Name -> callable dispatch table used by the agent loop.
TOOL_FUNCTIONS = {
    "run_command": run_command,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file_patch": edit_file_patch,
    "list_directory": list_directory,
    "get_file_outline": get_file_outline,
    "search_grep": search_grep,
}


# --------------------------------------------------------------------------
# Tool declarations for the Gemini API (explicit JSON schemas — no reliance
# on SDK introspection, so behaviour is identical across SDK versions)
# --------------------------------------------------------------------------

def _build_tools(types: Any) -> list:
    """Build the google.genai Tool object exposing our five functions."""
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="run_command",
                    description=(
                        "Execute a shell command in the Termux environment "
                        "and return its combined stdout/stderr plus exit "
                        "code. The user is asked to confirm every command "
                        "interactively before it runs, and a refusal is not "
                        "an error — adapt. Never use this for interactive "
                        "programs (vim, nano, top, REPLs) or for commands "
                        "that run forever."
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "command": {
                                "type": "STRING",
                                "description": (
                                    "The exact shell command to execute, "
                                    "e.g. 'python3 app.py' or "
                                    "'pkg install -y git'."
                                ),
                            },
                        },
                        "required": ["command"],
                    },
                ),
                types.FunctionDeclaration(
                    name="read_file",
                    description=(
                        "Read a UTF-8 text file and return its full content "
                        "(long files are truncated in the middle). Always "
                        "read a file before patching it."
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "path": {
                                "type": "STRING",
                                "description": "Path to the file (~ and "
                                               "relative paths are OK).",
                            },
                        },
                        "required": ["path"],
                    },
                ),
                types.FunctionDeclaration(
                    name="write_file",
                    description=(
                        "Write a new file or completely overwrite an "
                        "existing one with the given content. Parent "
                        "directories are created automatically. Prefer "
                        "edit_file_patch for small changes to large files."
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "path": {
                                "type": "STRING",
                                "description": "Path of the file to write.",
                            },
                            "content": {
                                "type": "STRING",
                                "description": "The complete file content.",
                            },
                        },
                        "required": ["path", "content"],
                    },
                ),
                types.FunctionDeclaration(
                    name="edit_file_patch",
                    description=(
                        "Patch a file by replacing exactly one occurrence of "
                        "search_block with replace_block. The search_block "
                        "must match the file content EXACTLY (every space "
                        "and newline) and occur exactly once — copy it from "
                        "a fresh read_file call. Use this instead of "
                        "rewriting large files."
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "path": {
                                "type": "STRING",
                                "description": "Path of the file to patch.",
                            },
                            "search_block": {
                                "type": "STRING",
                                "description": "Exact text to find (must "
                                               "match once, verbatim).",
                            },
                            "replace_block": {
                                "type": "STRING",
                                "description": "Replacement text.",
                            },
                        },
                        "required": ["path", "search_block", "replace_block"],
                    },
                ),
                types.FunctionDeclaration(
                    name="list_directory",
                    description=(
                        "List files and folders of a directory to understand "
                        "the project workspace. Directories are shown first "
                        "with a trailing slash; files include their size."
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "path": {
                                "type": "STRING",
                                "description": "Directory to list "
                                               "(default: current directory).",
                            },
                        },
                        "required": [],
                    },
                ),
                types.FunctionDeclaration(
                    name="get_file_outline",
                    description=(
                        "Extract a structural outline (classes, methods, functions) "
                        "from a Python source file using AST. Returns a clean hierarchy "
                        "with line numbers and signatures, helping to understand large files "
                        "without reading them fully."
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "path": {
                                "type": "STRING",
                                "description": "Path to the Python file.",
                            },
                        },
                        "required": ["path"],
                    },
                ),
                types.FunctionDeclaration(
                    name="search_grep",
                    description=(
                        "Search for a text pattern or regex query recursively across files "
                        "in a directory. Filters binary files and standard build/config "
                        "folders automatically. Returns matching line numbers and content."
                    ),
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "The text pattern or regex to search for.",
                            },
                            "path": {
                                "type": "STRING",
                                "description": "Directory to search from (default: current directory).",
                            },
                            "extension_filter": {
                                "type": "STRING",
                                "description": "Comma-separated list of extensions to search, e.g. 'py,js,json'.",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ]
        )
    ]


# --------------------------------------------------------------------------
# System instruction: defines the agent's behaviour and self-healing loop
# --------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are Ctx0an, an autonomous senior software engineer running on the user's Android device inside Termux (a Linux terminal environment). You do not just chat — you pair-program with the user by actually doing the work: exploring, reading, writing, patching and executing code through the provided tools.

# Operating loop
Always follow a strict Think -> Act -> Observe cycle:
1. THINK: reason briefly about the goal and the next step.
2. ACT: call the tool(s) for that step.
3. OBSERVE: study the tool result carefully before deciding the next step. Never ignore or assume tool output.

# Environment facts (Termux)
- No root and no sudo. Install system packages with `pkg install <name>` (or `apt`).
- Python is `python` / `python3`. Install Python packages with `pip install <name>`.
- Home is $HOME under /data/data/com.termux/files/home. Shared device storage (after `termux-setup-storage`) is ~/storage.
- Common tools (git, clang, node) may be missing — check and install them with pkg when a task needs them.

# Tool-use rules
1. Explore before acting: use list_directory to list workspace files. Use search_grep to find text/regex patterns across the codebase, and get_file_outline to view class/method skeletons of Python files without reading the entire file. Use read_file to view contents of specific target files.
2. Use write_file to create new files or fully rewrite small ones. Use edit_file_patch for targeted changes to existing files — the search_block must match the file EXACTLY, indentation included. If a patch fails, study the returned diagnostic diff carefully and edit your search_block.
3. run_command requires interactive user confirmation. If the user declines a command, do NOT repeat the same command — propose an alternative or ask.
4. Never start interactive programs (vim, nano, top, bare REPLs) or commands designed to run forever. Start servers only when the task explicitly requires it.
5. Prefer project-local, reversible changes. Never delete or overwrite anything outside the current workspace unless the user explicitly asked.

# Staged files (pinned context)
- The user may stage files to be pinned into your context. Staged files are formatted inside `### STAGED FILES CONTEXT` in the user message. Use them to understand current file contents without calling read_file again, but always confirm the latest state if you suspect they have changed.

# Self-healing behaviour
When you write or modify code, verify it: run it, or at least byte-compile it (`python3 -m py_compile file.py`). If a command fails (non-zero exit code, Traceback, compiler error):
1. Read the error output carefully.
2. Diagnose the root cause.
3. Fix it with edit_file_patch (or write_file for a full rewrite).
4. Run it again.
Repeat until it works, or until you can explain precisely why it cannot work in this environment. Never stop at the first error.

# Communication style
- Be concise. Briefly state what you are doing and why before each action (one or two sentences).
- When the task is complete, summarize: what changed, in which files, and how to run or use the result. Do not paste whole files back.
- If you are stuck after several attempts, say so clearly and ask the user one specific question."""


# --------------------------------------------------------------------------
# Gemini client loading (lazy, so --help and tool testing work without the SDK)
# --------------------------------------------------------------------------

def _load_genai() -> tuple[Any, Any]:
    """Import the google-genai SDK, exiting with guidance if it is missing."""
    try:
        from google import genai
        from google.genai import types
        return genai, types
    except ImportError:
        ui.error(
            "the 'google-genai' package is not installed.\n"
            "Install it with:\n"
            "    pip install google-genai\n"
            "(on Termux: pkg install python && pip install google-genai)"
        )
        sys.exit(2)


# --------------------------------------------------------------------------
# Model Context Protocol (MCP) Stdio JSON-RPC Client Integration
# --------------------------------------------------------------------------

def _get_mcp_config_file() -> Path:
    cfg_dir = Path.home() / ".ctx0an"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "mcp_config.json"
    if not cfg_file.exists():
        template = {
            "mcpServers": {
                "example-sqlite": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", str(cfg_dir / "sqlite.db")]
                }
            }
        }
        try:
            cfg_file.write_text(json.dumps(template, indent=2), encoding="utf-8")
        except Exception:
            pass
    return cfg_file


class MCPSession:
    def __init__(self, name: str, command: str, args: list[str]):
        self.name = name
        self.command = command
        self.args = args
        self.proc = None
        self.read_thread = None
        self.request_id = 0
        self.pending_requests = {}
        self.tools = []
        self.active = False

    def start(self):
        try:
            self.proc = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self.active = True
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            threading.Thread(target=self._stderr_loop, daemon=True).start()
        except Exception as e:
            ui.error(f"Failed to start MCP server '{self.name}': {e}")
            self.active = False

    def _read_loop(self):
        while self.active and self.proc:
            try:
                line = self.proc.stdout.readline()
                if not line:
                    break
                data = json.loads(line)
                if "id" in data:
                    req_id = data["id"]
                    if req_id in self.pending_requests:
                        event, holder = self.pending_requests[req_id]
                        holder["response"] = data
                        event.set()
            except Exception:
                break
        self.active = False

    def _stderr_loop(self):
        while self.active and self.proc:
            try:
                line = self.proc.stderr.readline()
                if not line:
                    break
            except Exception:
                break

    def call_method(self, method: str, params: dict | None = None, timeout: float = 10.0) -> dict | None:
        if not self.active or not self.proc:
            return None
        self.request_id += 1
        req_id = self.request_id
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method
        }
        if params is not None:
            payload["params"] = params
            
        event = threading.Event()
        holder = {"response": None}
        self.pending_requests[req_id] = (event, holder)
        
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except Exception:
            self.active = False
            return None
            
        if event.wait(timeout):
            del self.pending_requests[req_id]
            return holder["response"]
        else:
            if req_id in self.pending_requests:
                del self.pending_requests[req_id]
            return None

    def send_notification(self, method: str, params: dict | None = None):
        if not self.active or not self.proc:
            return
        payload = {
            "jsonrpc": "2.0",
            "method": method
        }
        if params is not None:
            payload["params"] = params
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except Exception:
            self.active = False

    def initialize(self) -> bool:
        init_res = self.call_method("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ctx0an-client", "version": "1.0.0"}
        })
        if not init_res or "error" in init_res:
            return False
            
        self.send_notification("notifications/initialized")
        
        tools_res = self.call_method("tools/list")
        if tools_res and "result" in tools_res:
            self.tools = tools_res["result"].get("tools", [])
            return True
        return False

    def call_tool(self, name: str, arguments: dict) -> str:
        res = self.call_method("tools/call", {
            "name": name,
            "arguments": arguments
        }, timeout=COMMAND_TIMEOUT)
        if not res:
            return "Error: MCP server timeout or connection lost."
        if "error" in res:
            return f"Error: MCP server error: {res['error']}"
        result_data = res.get("result", {})
        content = result_data.get("content", [])
        out = []
        for item in content:
            if item.get("type") == "text":
                out.append(item.get("text", ""))
        return "\n".join(out)

    def stop(self):
        self.active = False
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


class MCPManager:
    def __init__(self):
        self.sessions: list[MCPSession] = []
        self.tool_to_session: dict[str, MCPSession] = {}

    def load_and_initialize(self) -> list[dict]:
        config_path = _get_mcp_config_file()
        if not config_path.exists():
            return []
            
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            ui.warn(f"Failed to read MCP config from {config_path}: {e}")
            return []
            
        servers_config = config.get("mcpServers", {})
        if not servers_config:
            return []
            
        ui.info(f"Loading {len(servers_config)} MCP servers...")
        
        mcp_declarations = []
        
        for name, cfg in servers_config.items():
            cmd = cfg.get("command")
            if not cmd:
                continue
            args = cfg.get("args", [])
            
            session = MCPSession(name, cmd, args)
            session.start()
            if session.initialize():
                self.sessions.append(session)
                ui.success(f"MCP server '{name}' initialized with {len(session.tools)} tools.")
                for t in session.tools:
                    self.tool_to_session[t["name"]] = session
                    mcp_declarations.append(t)
            else:
                ui.error(f"Failed to initialize MCP server '{name}'.")
                session.stop()
                
        return mcp_declarations

    def close_all(self):
        for s in self.sessions:
            s.stop()
        self.sessions.clear()


def _mcp_schema_to_gemini(mcp_schema: dict, types: Any) -> Any:
    typ = mcp_schema.get("type", "object").upper()
    gemini_type = getattr(types.Type, typ, types.Type.OBJECT)
    
    properties = {}
    for prop_name, prop_val in mcp_schema.get("properties", {}).items():
        prop_type = prop_val.get("type", "string").upper()
        p_gemini_type = getattr(types.Type, prop_type, types.Type.STRING)
        properties[prop_name] = types.Schema(
            type=p_gemini_type,
            description=prop_val.get("description", "")
        )
        
    return types.Schema(
        type=gemini_type,
        properties=properties,
        required=mcp_schema.get("required", [])
    )


# --------------------------------------------------------------------------
# Multi-API Standard Library REST Clients (Anthropic Claude & OpenAI GPT)
# --------------------------------------------------------------------------

def _call_anthropic_api(model: str, system: str, messages: list[dict], tools: list[dict], api_key: str, base_url: str = None) -> dict:
    import urllib.request
    import json
    
    url = (base_url.rstrip("/") + "/messages") if base_url else "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    anthropic_tools = []
    for t in tools:
        anthropic_tools.append({
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"]
        })
        
    payload = {
        "model": model,
        "max_tokens": 4000,
        "system": system,
        "messages": messages,
    }
    if anthropic_tools:
        payload["tools"] = anthropic_tools

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=COMMAND_TIMEOUT) as response:
            res_body = response.read().decode("utf-8")
            return json.loads(res_body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
            err_msg = err_json.get("error", {}).get("message", err_body)
        except Exception:
            err_msg = err_body
        raise Exception(f"Anthropic API error ({e.code}): {err_msg}")


def _call_openai_api(model: str, system: str, messages: list[dict], tools: list[dict], api_key: str, base_url: str = None) -> dict:
    import urllib.request
    import json
    
    url = (base_url.rstrip("/") + "/chat/completions") if base_url else "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json"
    }
    
    openai_messages = [{"role": "system", "content": system}] + messages
    
    openai_tools = []
    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
        })
        
    payload = {
        "model": model,
        "messages": openai_messages,
    }
    if openai_tools:
        payload["tools"] = openai_tools

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=COMMAND_TIMEOUT) as response:
            res_body = response.read().decode("utf-8")
            return json.loads(res_body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
            err_msg = err_json.get("error", {}).get("message", err_body)
        except Exception:
            err_msg = err_body
        raise Exception(f"OpenAI API error ({e.code}): {err_msg}")


def _history_to_anthropic(history_dicts: list[dict]) -> list[dict]:
    messages = []
    for turn in history_dicts:
        role = "user" if turn["role"] == "user" else "assistant"
        content = []
        for part in turn.get("parts", []):
            if "text" in part:
                content.append({"type": "text", "text": part["text"]})
            elif "function_call" in part:
                call = part["function_call"]
                content.append({
                    "type": "tool_use",
                    "id": f"call_{call['name']}_{len(messages)}",
                    "name": call["name"],
                    "input": call["args"]
                })
            elif "function_response" in part:
                resp = part["function_response"]
                tool_use_id = None
                for m in reversed(messages):
                    if m["role"] == "assistant":
                        for item in m["content"]:
                            if item["type"] == "tool_use" and item["name"] == resp["name"]:
                                tool_use_id = item["id"]
                                break
                    if tool_use_id:
                        break
                if not tool_use_id:
                    tool_use_id = f"call_{resp['name']}_fallback"
                content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": str(resp["response"].get("result", ""))
                })
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _history_to_openai(history_dicts: list[dict]) -> list[dict]:
    messages = []
    for turn in history_dicts:
        role = turn["role"]
        if role == "model":
            role = "assistant"
            
        is_tool_response = False
        for part in turn.get("parts", []):
            if "function_response" in part:
                is_tool_response = True
                break
                
        if is_tool_response:
            for part in turn.get("parts", []):
                if "function_response" in part:
                    resp = part["function_response"]
                    tool_call_id = None
                    for m in reversed(messages):
                        if m["role"] == "assistant" and "tool_calls" in m:
                            for tc in m["tool_calls"]:
                                if tc["function"]["name"] == resp["name"]:
                                    tool_call_id = tc["id"]
                                    break
                        if tool_call_id:
                            break
                    if not tool_call_id:
                        tool_call_id = f"call_{resp['name']}_fallback"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": str(resp["response"].get("result", ""))
                    })
            continue
            
        content_text = ""
        tool_calls = []
        for part in turn.get("parts", []):
            if "text" in part:
                content_text += part["text"]
            elif "function_call" in part:
                call = part["function_call"]
                tc_id = f"call_{call['name']}_{len(messages)}"
                tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["args"])
                    }
                })
        
        msg = {"role": role}
        if content_text:
            msg["content"] = content_text
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)
    return messages


def _wrap_anthropic_response(res: dict, types: Any) -> Any:
    content_blocks = res.get("content", [])
    parts = []
    text_content = ""
    for block in content_blocks:
        if block.get("type") == "text":
            text_content += block.get("text", "")
        elif block.get("type") == "tool_use":
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=block["name"],
                        args=block["input"]
                    )
                )
            )
    if text_content:
        parts.insert(0, types.Part(text=text_content))
        
    class DummyUsage:
        def __init__(self, usage):
            self.prompt_token_count = usage.get("input_tokens", 0)
            self.candidates_token_count = usage.get("output_tokens", 0)
            self.total_token_count = self.prompt_token_count + self.candidates_token_count
            
    class DummyCandidate:
        def __init__(self, parts_list):
            self.content = types.Content(role="model", parts=parts_list)
            
    class DummyResponse:
        def __init__(self, cand_list, usage):
            self.candidates = cand_list
            self.usage_metadata = usage
            
    usage_data = res.get("usage", {})
    return DummyResponse([DummyCandidate(parts)], DummyUsage(usage_data))


def _wrap_openai_response(res: dict, types: Any) -> Any:
    choice = res.get("choices", [{}])[0]
    message = choice.get("message", {})
    parts = []
    
    text_content = message.get("content", "")
    if text_content:
        parts.append(types.Part(text=text_content))
        
    for tc in message.get("tool_calls", []):
        try:
            args = json.loads(tc["function"]["arguments"])
        except Exception:
            args = {}
        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    name=tc["function"]["name"],
                    args=args
                )
            )
        )
        
    class DummyUsage:
        def __init__(self, usage):
            self.prompt_token_count = usage.get("prompt_tokens", 0)
            self.candidates_token_count = usage.get("completion_tokens", 0)
            self.total_token_count = usage.get("total_tokens", 0)
            
    class DummyCandidate:
        def __init__(self, parts_list):
            self.content = types.Content(role="model", parts=parts_list)
            
    class DummyResponse:
        def __init__(self, cand_list, usage):
            self.candidates = cand_list
            self.usage_metadata = usage
            
    usage_data = res.get("usage", {})
    return DummyResponse([DummyCandidate(parts)], DummyUsage(usage_data))


# --------------------------------------------------------------------------
# The agent: owns the conversation history and the Think->Act->Observe loop
# --------------------------------------------------------------------------

class Ctx0anAgent:
    """Autonomous coding agent driven by manual Gemini function calling."""

    def __init__(self, model_name: str, api_key: str) -> None:
        genai, types = _load_genai()
        self._types = types
        self.model_name = model_name
        self._api_key = api_key
        self.client = genai.Client(api_key=api_key)
        
        # Initialize MCP Manager and load tools
        self.mcp_manager = MCPManager()
        mcp_tools = self.mcp_manager.load_and_initialize()
        
        agent_tools = _build_tools(types)
        for t in mcp_tools:
            try:
                gemini_tool = types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=_mcp_schema_to_gemini(t.get("inputSchema", {}), types)
                )
                agent_tools[0].function_declarations.append(gemini_tool)
            except Exception as e:
                ui.warn(f"Failed to register MCP tool '{t['name']}': {e}")
                
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=agent_tools,
            temperature=0.3,
        )
        # Full conversation history (user + model + function-response turns).
        self.history: list = []
        # Staged files for pinned context
        self.staged_files: set[str] = set()

    # -- API call with spinner, retries and graceful error handling ---------

    def _generate(self) -> Any:
        provider = "gemini"
        api_key_env = "GEMINI_API_KEY"
        base_url = None
        model_id = self.model_name
        
        custom_cfg = CUSTOM_MODELS.get(self.model_name)
        if custom_cfg:
            provider = custom_cfg.get("provider", "openai")
            default_env = "GEMINI_API_KEY" if provider == "gemini" else ("CLAUDE_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY")
            api_key_env = custom_cfg.get("api_key_env", default_env)
            base_url = custom_cfg.get("base_url")
            model_id = custom_cfg.get("model_id", self.model_name)
        else:
            provider = MODEL_PROVIDERS.get(self.model_name, "gemini")
            if provider == "gemini":
                api_key_env = "GEMINI_API_KEY"
            elif provider == "anthropic":
                api_key_env = "CLAUDE_API_KEY"
            elif provider == "openai":
                api_key_env = "OPENAI_API_KEY"

        if api_key_env == "CLAUDE_API_KEY":
            api_key = (self._api_key or os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")).strip()
        else:
            api_key = (self._api_key or os.environ.get(api_key_env, "")).strip()
            
        if not api_key:
            raise Exception(f"API key environment variable {api_key_env} is not set.")
            
        if provider == "gemini":
            last_exc: Exception | None = None
            tokens_info = ""
            genai, _ = _load_genai()
            self.client = genai.Client(api_key=api_key)
            try:
                token_count_resp = self.client.models.count_tokens(
                    model=model_id,
                    contents=self.history,
                )
                input_tokens = getattr(token_count_resp, "total_tokens", 0)
                tokens_info = f" | {input_tokens} context tokens"
            except Exception:
                pass

            for attempt in range(1, API_MAX_RETRIES + 1):
                try:
                    with ui.status(f"thinking with {model_id}{tokens_info} ..."):
                        return self.client.models.generate_content(
                            model=model_id,
                            contents=self.history,
                            config=self.config,
                        )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    last_exc = exc
                    code = (getattr(exc, "code", None) or getattr(exc, "status_code", None))
                    if code in (400, 401, 403, 404):
                        raise
                    ui.warn(f"API error (attempt {attempt}/{API_MAX_RETRIES}): {exc}")
                    if attempt < API_MAX_RETRIES:
                        time.sleep(min(2 ** attempt, 8))
            raise last_exc

        elif provider in ("anthropic", "openai"):
            history_dicts = json.loads(_serialize_history(self.history, self._types))
            
            raw_tools = []
            if self.config.tools and self.config.tools[0].function_declarations:
                for dec in self.config.tools[0].function_declarations:
                    p_type = str(dec.parameters.type).split(".")[-1].lower() if hasattr(dec.parameters, "type") else "object"
                    properties = {}
                    if hasattr(dec.parameters, "properties") and dec.parameters.properties:
                        for p_name, p_val in dec.parameters.properties.items():
                            val_type = str(p_val.type).split(".")[-1].lower() if hasattr(p_val, "type") else "string"
                            properties[p_name] = {
                                "type": val_type,
                                "description": getattr(p_val, "description", "")
                            }
                    raw_tools.append({
                        "name": dec.name,
                        "description": dec.description,
                        "parameters": {
                            "type": p_type,
                            "properties": properties,
                            "required": getattr(dec.parameters, "required", []) or []
                        }
                    })
                
            last_exc = None
            for attempt in range(1, API_MAX_RETRIES + 1):
                try:
                    with ui.status(f"thinking with {self.model_name} ..."):
                        if provider == "anthropic":
                            anthropic_messages = _history_to_anthropic(history_dicts)
                            res = _call_anthropic_api(model_id, SYSTEM_INSTRUCTION, anthropic_messages, raw_tools, api_key, base_url)
                            return _wrap_anthropic_response(res, self._types)
                        else:
                            openai_messages = _history_to_openai(history_dicts)
                            res = _call_openai_api(model_id, SYSTEM_INSTRUCTION, openai_messages, raw_tools, api_key, base_url)
                            return _wrap_openai_response(res, self._types)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    last_exc = exc
                    ui.warn(f"API error (attempt {attempt}/{API_MAX_RETRIES}): {exc}")
                    if attempt < API_MAX_RETRIES:
                        time.sleep(min(2 ** attempt, 8))
            raise last_exc

    # -- tool dispatch --------------------------------------------------------

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        ui.tool_call(name, args)
        fn = TOOL_FUNCTIONS.get(name)
        if fn is None:
            # Check if this tool is registered to an active MCP session
            if name in self.mcp_manager.tool_to_session:
                session = self.mcp_manager.tool_to_session[name]
                try:
                    return session.call_tool(name, args)
                except Exception as exc:
                    return f"Error: MCP tool call failed: {exc}"
            return (f"Error: unknown tool '{name}'. Available tools: "
                    f"{', '.join(sorted(list(TOOL_FUNCTIONS.keys()) + list(self.mcp_manager.tool_to_session.keys())))}.")
        try:
            result = fn(**args)
        except TypeError as exc:
            return f"Error: invalid arguments for {name}: {exc}"
        except Exception as exc:
            return f"Error: {name} failed with {type(exc).__name__}: {exc}"
        if isinstance(result, (list, dict)):
            try:
                return json.dumps(result, ensure_ascii=False)
            except Exception:
                return str(result)
        return str(result)

    # -- main loop: one user message -> as many tool steps as needed ---------

    def run_turn(self, user_message: str) -> None:
        types = self._types
        
        # Build staged files context block
        staged_context = ""
        if self.staged_files:
            staged_context = "### STAGED FILES CONTEXT\n"
            staged_context += "The following files are pinned context. These are the current file contents:\n\n"
            for f_path in sorted(self.staged_files):
                try:
                    p = _resolve_path(f_path)
                    if p.exists() and p.is_file():
                        content = p.read_text(encoding="utf-8", errors="replace")
                        staged_context += f"--- START FILE: {os.path.relpath(p)} ---\n"
                        staged_context += content
                        staged_context += f"\n--- END FILE: {os.path.relpath(p)} ---\n\n"
                    else:
                        staged_context += f"--- FILE NOT FOUND: {f_path} ---\n\n"
                except Exception as e:
                    staged_context += f"--- ERROR READING FILE {f_path}: {e} ---\n\n"
            staged_context += "### END OF STAGED FILES CONTEXT\n\n"
            
        full_message = staged_context + user_message
        self.history.append(
            types.Content(role="user",
                          parts=[types.Part(text=full_message)])
        )
        
        for _step in range(MAX_AGENT_STEPS):
            response = self._generate()

            candidates = getattr(response, "candidates", None) or []
            content = (getattr(candidates[0], "content", None)
                       if candidates else None)
            if content is None:
                feedback = getattr(response, "prompt_feedback", None)
                ui.error(f"the model returned no content. {feedback or ''}")
                return
            self.history.append(content)

            parts = content.parts or []
            text = "".join(
                p.text for p in parts if getattr(p, "text", None)
            )
            if text.strip():
                ui.assistant(text)

            # Report token usage and cost estimation
            usage = getattr(response, "usage_metadata", None)
            if usage:
                prompt_tok = getattr(usage, "prompt_token_count", 0)
                cand_tok = getattr(usage, "candidates_token_count", 0)
                tot_tok = getattr(usage, "total_token_count", 0)
                cost_str = ""
                if "gemini-2.5-pro" in self.model_name:
                    cost = (prompt_tok * 1.25 + cand_tok * 5.00) / 1_000_000
                    cost_str = f" (~${cost:.5f})"
                elif "gemini-2.5-flash" in self.model_name:
                    cost = (prompt_tok * 0.075 + cand_tok * 0.30) / 1_000_000
                    cost_str = f" (~${cost:.5f})"
                elif "claude-3-5-sonnet" in self.model_name:
                    cost = (prompt_tok * 3.00 + cand_tok * 15.00) / 1_000_000
                    cost_str = f" (~${cost:.5f})"
                elif "claude-3-5-haiku" in self.model_name:
                    cost = (prompt_tok * 0.25 + cand_tok * 1.25) / 1_000_000
                    cost_str = f" (~${cost:.5f})"
                elif "gpt-4o-mini" in self.model_name:
                    cost = (prompt_tok * 0.15 + cand_tok * 0.60) / 1_000_000
                    cost_str = f" (~${cost:.5f})"
                elif "gpt-4o" in self.model_name:
                    cost = (prompt_tok * 2.50 + cand_tok * 10.00) / 1_000_000
                    cost_str = f" (~${cost:.5f})"
                ui.info(f"[usage: {prompt_tok} in, {cand_tok} out, {tot_tok} total{cost_str}]")

            calls = [
                p.function_call for p in parts
                if getattr(p, "function_call", None)
                and getattr(p.function_call, "name", None)
            ]
            if not calls:
                return                      # final answer reached

            # Execute every requested tool and feed the results back.
            fn_response_parts = []
            for call in calls:
                try:
                    args = dict(call.args or {})
                except Exception:
                    args = {}
                result = self._execute_tool(call.name, args)
                fn_response_parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result},
                    )
                )
            self.history.append(
                types.Content(role="user", parts=fn_response_parts)
            )

        ui.warn(f"stopped after {MAX_AGENT_STEPS} consecutive tool steps "
                "(safety limit). Ask the agent to continue if needed.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

HELP_TEXT = """Interactive session commands:
    /help            show this help
    /model           show the current model
    /model flash     switch model (pro | flash | any Gemini model id)
    /status          show the current session status (including staged files)
    /new, /clear     start a new session / forget conversation history
    /add <path>      stage a file (pin it to the active conversation context)
    /drop <path>     unstage/drop a file from active context
    /staged          list all staged files
    /save <name>     save conversation history to a named session
    /load <name>     load a past conversation session
    /sessions        list all saved conversation sessions
    /exit, exit      leave the session (Ctrl-D also works)
Anything else is sent to the agent as a task."""


def _get_custom_models() -> dict:
    config_dir = Path.home() / ".ctx0an"
    config_dir.mkdir(parents=True, exist_ok=True)
    models_file = config_dir / "models.json"
    
    if not models_file.exists():
        default_data = {
            "custom_models": {
                "deepseek-coder": {
                    "provider": "openai",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "model_id": "deepseek-coder",
                    "description": "DeepSeek Coder (OpenAI Protocol)"
                },
                "groq-llama": {
                    "provider": "openai",
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key_env": "GROQ_API_KEY",
                    "model_id": "llama-3.3-70b-versatile",
                    "description": "Groq Llama 3.3 (OpenAI Protocol)"
                },
                "ollama-codellama": {
                    "provider": "openai",
                    "base_url": "http://localhost:11434/v1",
                    "api_key_env": "OLLAMA_API_KEY",
                    "model_id": "codellama",
                    "description": "Local Ollama CodeLlama"
                }
            }
        }
        try:
            models_file.write_text(json.dumps(default_data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return default_data.get("custom_models", {})
        
    try:
        data = json.loads(models_file.read_text(encoding="utf-8"))
        return data.get("custom_models", {})
    except Exception:
        return {}


CUSTOM_MODELS = _get_custom_models()


def _get_session_dir() -> Path:
    session_dir = Path.home() / ".ctx0an" / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _serialize_history(history: list, types: Any) -> str:
    serializable = []
    for content in history:
        parts_list = []
        for part in (content.parts or []):
            part_dict = {}
            if getattr(part, "text", None) is not None:
                part_dict["text"] = part.text
            elif getattr(part, "function_call", None) is not None:
                call = part.function_call
                part_dict["function_call"] = {
                    "name": call.name,
                    "args": dict(call.args or {})
                }
            elif getattr(part, "function_response", None) is not None:
                resp = part.function_response
                part_dict["function_response"] = {
                    "name": resp.name,
                    "response": dict(resp.response or {})
                }
            parts_list.append(part_dict)
        serializable.append({
            "role": content.role,
            "parts": parts_list
        })
    return json.dumps(serializable, indent=2, ensure_ascii=False)


def _deserialize_history(json_str: str, types: Any) -> list:
    history = []
    data = json.loads(json_str)
    for content_dict in data:
        parts = []
        for part_dict in content_dict["parts"]:
            if "text" in part_dict:
                parts.append(types.Part(text=part_dict["text"]))
            elif "function_call" in part_dict:
                call_data = part_dict["function_call"]
                parts.append(types.Part(
                    function_call=types.FunctionCall(
                        name=call_data["name"],
                        args=call_data["args"]
                    )
                ))
            elif "function_response" in part_dict:
                resp_data = part_dict["function_response"]
                parts.append(
                    types.Part.from_function_response(
                        name=resp_data["name"],
                        response=resp_data["response"]
                    )
                )
        history.append(types.Content(role=content_dict["role"], parts=parts))
    return history


def resolve_model(name: str) -> str:
    """Map friendly aliases to full model ids; pass unknown ids through."""
    cleaned = (name or "").strip()
    if not cleaned:
        return DEFAULT_MODEL
    if cleaned in CUSTOM_MODELS:
        return cleaned
    return MODEL_ALIASES.get(cleaned.lower(), cleaned)


# --------------------------------------------------------------------------
# Web GUI Server and HTML content (100% standard library, Termux compatible)
# --------------------------------------------------------------------------

GUI_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ctx0an Web Chat UI</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0d0f12;
            --panel-bg: rgba(20, 24, 30, 0.7);
            --sidebar-bg: #11141a;
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.15);
            --text: #e2e8f0;
            --text-dim: #94a3b8;
            --primary: #00f0ff;
            --primary-glow: rgba(0, 240, 255, 0.15);
            --accent: #f0abfc;
            --success: #10b981;
            --warn: #f59e0b;
            --danger: #ef4444;
            --user-bubble: #1e293b;
            --bot-bubble: rgba(30, 41, 59, 0.4);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        /* Sidebar styling */
        .sidebar {
            width: 320px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            z-index: 10;
        }

        .brand {
            padding: 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand h1 {
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand span {
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-dim);
        }

        .section-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-dim);
            padding: 20px 24px 8px;
            font-weight: 600;
        }

        .staged-files, .sessions-list {
            flex: 1;
            overflow-y: auto;
            padding: 0 16px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .file-item, .session-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 14px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 13px;
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .file-item:hover, .session-item:hover {
            border-color: var(--border-hover);
            background: rgba(255, 255, 255, 0.04);
        }

        .file-item.selected, .session-item.selected {
            border-color: var(--primary);
            background: var(--primary-glow);
        }

        .file-name, .session-name {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 200px;
        }

        .file-action, .session-action {
            color: var(--text-dim);
            background: none;
            border: none;
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            font-size: 11px;
        }

        .file-action:hover, .session-action:hover {
            color: var(--danger);
            background: rgba(239, 68, 68, 0.1);
        }

        .add-file-box, .session-save-box {
            padding: 16px;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 8px;
        }

        input {
            flex: 1;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 12px;
            color: var(--text);
            font-family: inherit;
            font-size: 13px;
            transition: all 0.2s ease;
        }

        input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }

        button.btn-primary {
            background: var(--primary);
            color: #000;
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        button.btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 240, 255, 0.3);
        }

        /* Main Area */
        .main-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            background-image: radial-gradient(circle at 50% 50%, rgba(20, 24, 30, 0.4) 0%, rgba(13, 15, 18, 0.9) 100%);
        }

        /* Header bar */
        .header-bar {
            height: 70px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 32px;
            backdrop-filter: blur(12px);
            z-index: 5;
        }

        .status-info {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 13px;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(0, 240, 255, 0.05);
            border: 1px solid rgba(0, 240, 255, 0.15);
            color: var(--primary);
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 500;
        }

        .token-info {
            color: var(--text-dim);
        }

        .controls {
            display: flex;
            gap: 12px;
        }

        .btn-clear {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            color: var(--text);
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-clear:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--border-hover);
        }

        /* Chat window */
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 32px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .message {
            display: flex;
            flex-direction: column;
            max-width: 85%;
            animation: fadeIn 0.3s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
        }

        .message.bot {
            align-self: flex-start;
        }

        .message-bubble {
            padding: 16px 20px;
            border-radius: 16px;
            font-size: 15px;
            line-height: 1.6;
            word-break: break-word;
        }

        .message.user .message-bubble {
            background-color: var(--user-bubble);
            border-bottom-right-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .message.bot .message-bubble {
            background-color: var(--bot-bubble);
            border-bottom-left-radius: 4px;
            border: 1px solid var(--border);
            backdrop-filter: blur(8px);
        }

        .message-sender {
            font-size: 11px;
            color: var(--text-dim);
            margin-bottom: 4px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .message.user .message-sender {
            align-self: flex-end;
        }

        /* Tool box styles */
        .tool-box {
            margin-top: 12px;
            border: 1px solid rgba(240, 171, 252, 0.2);
            background: rgba(240, 171, 252, 0.02);
            border-radius: 10px;
            overflow: hidden;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }

        .tool-header {
            padding: 10px 16px;
            background: rgba(240, 171, 252, 0.05);
            border-bottom: 1px solid rgba(240, 171, 252, 0.15);
            display: flex;
            justify-content: space-between;
            color: var(--accent);
            font-weight: 500;
        }

        .tool-body {
            padding: 12px 16px;
            white-space: pre-wrap;
            color: #d1d5db;
            max-height: 200px;
            overflow-y: auto;
        }

        .tool-result-box {
            margin-top: 8px;
            border: 1px solid var(--border);
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }

        .tool-result-header {
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border);
            color: var(--text-dim);
        }

        .tool-result-body {
            padding: 10px 12px;
            white-space: pre-wrap;
            color: #9ca3af;
            max-height: 150px;
            overflow-y: auto;
        }

        /* Confirmation box */
        .confirm-box {
            margin-top: 16px;
            background: rgba(245, 158, 11, 0.05);
            border: 1px dashed var(--warn);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            backdrop-filter: blur(8px);
        }

        .confirm-msg {
            font-size: 14px;
            color: #fef08a;
            font-weight: 500;
        }

        .confirm-actions {
            display: flex;
            gap: 12px;
        }

        .btn-confirm {
            padding: 10px 20px;
            border-radius: 6px;
            border: none;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-confirm.approve {
            background-color: var(--success);
            color: #fff;
        }

        .btn-confirm.reject {
            background-color: var(--danger);
            color: #fff;
        }

        .btn-confirm:hover {
            transform: translateY(-1px);
            filter: brightness(1.1);
        }

        /* Input Area */
        .input-container {
            padding: 24px 32px 32px;
            backdrop-filter: blur(12px);
            border-top: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .input-bar {
            display: flex;
            gap: 16px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 8px 12px;
            align-items: center;
        }

        .input-bar:focus-within {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }

        .input-bar textarea {
            flex: 1;
            background: none;
            border: none;
            resize: none;
            color: var(--text);
            font-family: inherit;
            font-size: 15px;
            height: 24px;
            padding-top: 2px;
        }

        .input-bar textarea:focus {
            outline: none;
        }

        .btn-send {
            background: var(--primary);
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #000;
            transition: all 0.2s ease;
        }

        .btn-send:hover {
            transform: scale(1.05);
            box-shadow: 0 0 10px var(--primary);
        }

        /* Markdown Styling */
        .message-bubble pre {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            margin: 10px 0;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }

        .message-bubble code {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(255, 255, 255, 0.06);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 13px;
        }

        .message-bubble pre code {
            background: none;
            padding: 0;
            font-size: 13px;
        }

        .message-bubble p {
            margin-bottom: 12px;
        }

        .message-bubble p:last-child {
            margin-bottom: 0;
        }

        .message-bubble ul, .message-bubble ol {
            margin-left: 20px;
            margin-bottom: 12px;
        }

        .message-bubble li {
            margin-bottom: 4px;
        }

        /* Spinner */
        .thinking-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 10px 16px;
            color: var(--text-dim);
            font-size: 13px;
        }

        .dot {
            width: 6px;
            height: 6px;
            background-color: var(--primary);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1.0); }
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="brand">
            <h1>Ctx0an UI</h1>
            <span id="version">v1.0.0</span>
        </div>

        <div class="section-title">Staged Files (Context)</div>
        <div class="staged-files" id="staged-files-list">
            <!-- Staged files here -->
        </div>
        <div class="add-file-box">
            <input type="text" id="add-file-input" placeholder="path/to/file.py">
            <button class="btn-primary" onclick="addStagedFile()">Stg</button>
        </div>

        <div class="section-title">Sessions</div>
        <div class="sessions-list" id="sessions-list-container">
            <!-- Saved sessions here -->
        </div>
        <div class="session-save-box">
            <input type="text" id="save-session-input" placeholder="session_name">
            <button class="btn-primary" onclick="saveSession()">Save</button>
        </div>
    </div>

    <!-- Main Container -->
    <div class="main-container">
        <!-- Header -->
        <div class="header-bar">
            <div class="status-info">
                <div class="status-badge" id="model-badge">gemini-2.5-pro</div>
                <div class="token-info" id="session-info">0 history items | 0 staged files</div>
            </div>
            <div class="controls">
                <button class="btn-clear" onclick="clearConversation()">Clear Chat</button>
            </div>
        </div>

        <!-- Chat Area -->
        <div class="chat-area" id="chat-window">
            <div class="message bot">
                <div class="message-sender">Ctx0an</div>
                <div class="message-bubble">
                    <p>Hello! I am Ctx0an, your autonomous software-development partner. How can I help you in your workspace today?</p>
                </div>
            </div>
        </div>

        <!-- Input Area -->
        <div class="input-container">
            <div class="input-bar">
                <textarea id="prompt-input" placeholder="Ask Ctx0an to write code, debug, or explore..." onkeydown="handleKeydown(event)"></textarea>
                <button class="btn-send" onclick="sendPrompt()">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                </button>
            </div>
        </div>
    </div>

    <script>
        let isThinking = false;
        let eventSource = null;
        let activeConfirmMsgId = null;

        const textarea = document.getElementById('prompt-input');
        textarea.addEventListener('input', () => {
            textarea.style.height = 'auto';
            textarea.style.height = (textarea.scrollHeight - 12) + 'px';
            if (textarea.scrollHeight > 150) {
                textarea.style.overflowY = 'auto';
            } else {
                textarea.style.overflowY = 'hidden';
            }
        });

        function handleKeydown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendPrompt();
            }
        }

        async function init() {
            await loadStatus();
            await loadStagedFiles();
            await loadSessions();
        }

        async function loadStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('model-badge').textContent = data.model;
                document.getElementById('session-info').textContent = `${data.history_len} items | ${data.staged_len} staged`;
            } catch (e) {
                console.error(e);
            }
        }

        async function loadStagedFiles() {
            try {
                const res = await fetch('/api/staged');
                const files = await res.json();
                const container = document.getElementById('staged-files-list');
                container.innerHTML = '';
                
                if (files.length === 0) {
                    container.innerHTML = '<div style="color:var(--text-dim);font-size:12px;padding:8px;">No files staged</div>';
                    return;
                }
                
                files.forEach(f => {
                    const item = document.createElement('div');
                    item.className = 'file-item';
                    
                    const nameSpan = document.createElement('span');
                    nameSpan.className = 'file-name';
                    nameSpan.textContent = f.name;
                    nameSpan.title = f.rel_path;
                    
                    const dropBtn = document.createElement('button');
                    dropBtn.className = 'file-action';
                    dropBtn.innerHTML = 'Drop';
                    dropBtn.onclick = () => dropStagedFile(f.path);
                    
                    item.appendChild(nameSpan);
                    item.appendChild(dropBtn);
                    container.appendChild(item);
                });
            } catch (e) {
                console.error(e);
            }
        }

        async function addStagedFile() {
            const input = document.getElementById('add-file-input');
            const path = input.value.trim();
            if (!path) return;
            try {
                const res = await fetch('/api/staged/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    input.value = '';
                    await loadStagedFiles();
                    await loadStatus();
                } else {
                    alert(data.error);
                }
            } catch (e) {
                alert('Failed to stage file: ' + e);
            }
        }

        async function dropStagedFile(path) {
            try {
                const res = await fetch('/api/staged/drop', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path })
                });
                await loadStagedFiles();
                await loadStatus();
            } catch (e) {
                alert('Failed to drop file: ' + e);
            }
        }

        async function loadSessions() {
            try {
                const res = await fetch('/api/sessions');
                const sessions = await res.json();
                const container = document.getElementById('sessions-list-container');
                container.innerHTML = '';
                
                if (sessions.length === 0) {
                    container.innerHTML = '<div style="color:var(--text-dim);font-size:12px;padding:8px;">No sessions saved</div>';
                    return;
                }
                
                sessions.forEach(s => {
                    const item = document.createElement('div');
                    item.className = 'session-item';
                    item.onclick = () => loadSession(s);
                    
                    const nameSpan = document.createElement('span');
                    nameSpan.className = 'session-name';
                    nameSpan.textContent = s;
                    
                    item.appendChild(nameSpan);
                    container.appendChild(item);
                });
            } catch (e) {
                console.error(e);
            }
        }

        async function saveSession() {
            const input = document.getElementById('save-session-input');
            const name = input.value.trim();
            if (!name) return;
            try {
                const res = await fetch('/api/sessions/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    input.value = '';
                    await loadSessions();
                } else {
                    alert(data.error);
                }
            } catch (e) {
                alert('Failed to save session: ' + e);
            }
        }

        async function loadSession(name) {
            if (!confirm(`Load session "${name}"? This will replace your current chat history.`)) return;
            try {
                const res = await fetch('/api/sessions/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    const chat = document.getElementById('chat-window');
                    chat.innerHTML = '';
                    await loadStatus();
                    appendBotMessage('Session loaded successfully. Ask a question to resume conversation.');
                }
            } catch (e) {
                alert('Failed to load session: ' + e);
            }
        }

        async function clearConversation() {
            if (!confirm('Clear all conversation history?')) return;
            try {
                await fetch('/api/clear', { method: 'POST' });
                const chat = document.getElementById('chat-window');
                chat.innerHTML = '';
                appendBotMessage('Conversation cleared. Workspace context remains active.');
                await loadStatus();
            } catch (e) {
                console.error(e);
            }
        }

        function appendUserMessage(text) {
            const chat = document.getElementById('chat-window');
            const msg = document.createElement('div');
            msg.className = 'message user';
            msg.innerHTML = `
                <div class="message-sender">you</div>
                <div class="message-bubble"><p>${escapeHtml(text)}</p></div>
            `;
            chat.appendChild(msg);
            scrollToBottom();
        }

        function appendBotMessage(htmlContent) {
            const chat = document.getElementById('chat-window');
            const msg = document.createElement('div');
            msg.className = 'message bot';
            msg.innerHTML = `
                <div class="message-sender">Ctx0an</div>
                <div class="message-bubble">${htmlContent}</div>
            `;
            chat.appendChild(msg);
            scrollToBottom();
            return msg;
        }

        function scrollToBottom() {
            const chat = document.getElementById('chat-window');
            chat.scrollTop = chat.scrollHeight;
        }

        function escapeHtml(text) {
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function renderMarkdown(text) {
            let html = escapeHtml(text);
            html = html.replace(/```(\\w*)\\n([\\s\\S]*?)```/g, (match, lang, code) => {
                return `<pre><code class="language-${lang}">${code}</code></pre>`;
            });
            html = html.replace(/`([^`\\n]+)`/g, '<code>$1</code>');
            html = html.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
            html = html.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
            html = html.replace(/\\n\\n/g, '</p><p>');
            html = html.replace(/\\n/g, '<br>');
            return `<p>${html}</p>`;
        }

        async function sendPrompt() {
            if (isThinking) return;
            const input = document.getElementById('prompt-input');
            const text = input.value.trim();
            if (!text) return;

            appendUserMessage(text);
            input.value = '';
            textarea.style.height = 'auto';

            isThinking = true;
            showThinkingIndicator();
            connectStream();

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();
                if (data.status !== 'ok') {
                    removeThinkingIndicator();
                    appendBotMessage(`<span style="color:var(--danger)">Error: ${escapeHtml(data.error)}</span>`);
                    isThinking = false;
                }
            } catch (e) {
                removeThinkingIndicator();
                appendBotMessage(`<span style="color:var(--danger)">Error: ${e}</span>`);
                isThinking = false;
            }
        }

        function showThinkingIndicator() {
            const chat = document.getElementById('chat-window');
            const indicator = document.createElement('div');
            indicator.className = 'thinking-indicator';
            indicator.id = 'thinking-indicator';
            indicator.innerHTML = `
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
                <span>thinking...</span>
            `;
            chat.appendChild(indicator);
            scrollToBottom();
        }

        function removeThinkingIndicator() {
            const ind = document.getElementById('thinking-indicator');
            if (ind) ind.remove();
        }

        function connectStream() {
            if (eventSource) {
                eventSource.close();
            }

            eventSource = new EventSource('/api/stream');
            let activeBotMsg = null;
            let responseText = '';

            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'text') {
                    removeThinkingIndicator();
                    if (!activeBotMsg) {
                        activeBotMsg = appendBotMessage('');
                    }
                    responseText += data.content;
                    activeBotMsg.querySelector('.message-bubble').innerHTML = renderMarkdown(responseText);
                    scrollToBottom();
                } 
                else if (data.type === 'tool_call') {
                    removeThinkingIndicator();
                    if (!activeBotMsg) {
                        activeBotMsg = appendBotMessage('');
                    }
                    const bubble = activeBotMsg.querySelector('.message-bubble');
                    const toolBox = document.createElement('div');
                    toolBox.className = 'tool-box';
                    const argsSummary = Object.entries(data.args || {}).map(([k, v]) => `${k}=\${JSON.stringify(v)}`).join(', ');
                    toolBox.innerHTML = `
                        <div class="tool-header">
                            <span>▸ tool: \${escapeHtml(data.name)}</span>
                        </div>
                        <div class="tool-body">\${escapeHtml(argsSummary || '(no args)')}</div>
                    `;
                    bubble.appendChild(toolBox);
                    scrollToBottom();
                } 
                else if (data.type === 'tool_result') {
                    if (activeBotMsg) {
                        const bubble = activeBotMsg.querySelector('.message-bubble');
                        const resultBox = document.createElement('div');
                        resultBox.className = 'tool-result-box';
                        let preview = data.content;
                        const lines = preview.split('\\n');
                        if (lines.length > 8) {
                            preview = lines.slice(0, 8).join('\\n') + `\\n... (\${lines.length - 8} more lines)`;
                        }
                        resultBox.innerHTML = `
                            <div class="tool-result-header">Result</div>
                            <div class="tool-result-body">\${escapeHtml(preview)}</div>
                        `;
                        bubble.appendChild(resultBox);
                        scrollToBottom();
                    }
                } 
                else if (data.type === 'confirm_request') {
                    removeThinkingIndicator();
                    if (!activeBotMsg) {
                        activeBotMsg = appendBotMessage('');
                    }
                    const bubble = activeBotMsg.querySelector('.message-bubble');
                    const confirmDiv = document.createElement('div');
                    confirmDiv.className = 'confirm-box';
                    confirmDiv.id = 'confirm-action-box';
                    confirmDiv.innerHTML = `
                        <div class="confirm-msg">\${escapeHtml(data.question)}</div>
                        <div class="confirm-actions">
                            <button class="btn-confirm approve" onclick="sendConfirmation(true)">Approve</button>
                            <button class="btn-confirm reject" onclick="sendConfirmation(false)">Reject</button>
                        </div>
                    `;
                    bubble.appendChild(confirmDiv);
                    scrollToBottom();
                } 
                else if (data.type === 'done') {
                    eventSource.close();
                    eventSource = null;
                    isThinking = false;
                    loadStatus();
                } 
                else if (data.type === 'error') {
                    removeThinkingIndicator();
                    appendBotMessage(`<span style="color:var(--danger)">Error: \${escapeHtml(data.content)}</span>`);
                    eventSource.close();
                    eventSource = null;
                    isThinking = false;
                    loadStatus();
                }
            };

            eventSource.onerror = (e) => {
                console.error('SSE Error:', e);
                eventSource.close();
                eventSource = null;
                isThinking = false;
            };
        }

        async function sendConfirmation(approved) {
            const confirmDiv = document.getElementById('confirm-action-box');
            if (confirmDiv) {
                confirmDiv.innerHTML = `<div class="confirm-msg" style="color:var(--text-dim)">\${approved ? 'Approved' : 'Rejected'}</div>`;
                confirmDiv.id = '';
            }
            try {
                await fetch('/api/confirm', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ approved })
                });
                showThinkingIndicator();
            } catch (e) {
                alert('Error: ' + e);
            }
        }

        window.onload = init;
    </script>
</body>
</html>'''

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

class Ctx0anHTTPRequestHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(GUI_HTML.encode("utf-8"))
        elif self.path == "/api/status":
            self.send_json({
                "model": global_agent.model_name if global_agent else DEFAULT_MODEL,
                "history_len": len(global_agent.history) if global_agent else 0,
                "staged_len": len(global_agent.staged_files) if global_agent else 0
            })
        elif self.path == "/api/staged":
            files = []
            if global_agent:
                for path_str in sorted(global_agent.staged_files):
                    p = Path(path_str)
                    files.append({
                        "name": p.name,
                        "path": path_str,
                        "rel_path": os.path.relpath(p)
                    })
            self.send_json(files)
        elif self.path == "/api/sessions":
            try:
                s_dir = _get_session_dir()
                files = sorted(s_dir.glob("*.json"))
                self.send_json([f.stem for f in files])
            except Exception:
                self.send_json([])
        elif self.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            while True:
                try:
                    event = gui_event_queue.get(timeout=1.0)
                    data_str = json.dumps(event)
                    self.wfile.write(f"data: {data_str}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if event.get("type") in ("done", "error"):
                        break
                except queue.Empty:
                    try:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    except Exception:
                        break
                except Exception:
                    break
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        try:
            params = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            params = {}

        if self.path == "/api/chat":
            message = params.get("message", "").strip()
            if not message:
                self.send_json({"error": "Empty message"}, status=400)
                return
            
            while not gui_event_queue.empty():
                try:
                    gui_event_queue.get_nowait()
                except queue.Empty:
                    break
                    
            def bg_turn():
                try:
                    if global_agent:
                        global_agent.run_turn(message)
                    gui_event_queue.put({"type": "done"})
                except Exception as e:
                    gui_event_queue.put({"type": "error", "content": str(e)})

            threading.Thread(target=bg_turn, daemon=True).start()
            self.send_json({"status": "ok"})
            
        elif self.path == "/api/confirm":
            approved = bool(params.get("approved", False))
            global gui_confirm_response
            gui_confirm_response = approved
            gui_confirm_event.set()
            self.send_json({"status": "ok"})
            
        elif self.path == "/api/staged/add":
            path = params.get("path", "").strip()
            if not path or not global_agent:
                self.send_json({"error": "Path required"}, status=400)
                return
            try:
                p = _resolve_path(path)
                if not p.exists():
                    self.send_json({"error": f"File not found: {path}"}, status=400)
                elif p.is_dir():
                    self.send_json({"error": f"'{path}' is a directory"}, status=400)
                else:
                    global_agent.staged_files.add(str(p))
                    self.send_json({"status": "ok"})
            except Exception as e:
                self.send_json({"error": str(e)}, status=500)
                
        elif self.path == "/api/staged/drop":
            path = params.get("path", "").strip()
            if not path or not global_agent:
                self.send_json({"error": "Path required"}, status=400)
                return
            if path in global_agent.staged_files:
                global_agent.staged_files.remove(path)
                self.send_json({"status": "ok"})
            else:
                self.send_json({"error": "File not staged"}, status=400)
                
        elif self.path == "/api/sessions/save":
            name = params.get("name", "").strip()
            name = re.sub(r"[^\w\-_]", "", name)
            if not name or not global_agent:
                self.send_json({"error": "Invalid name"}, status=400)
                return
            try:
                s_dir = _get_session_dir()
                filepath = s_dir / f"{name}.json"
                serialized = _serialize_history(global_agent.history, global_agent._types)
                filepath.write_text(serialized, encoding="utf-8")
                self.send_json({"status": "ok"})
            except Exception as e:
                self.send_json({"error": str(e)}, status=500)
                
        elif self.path == "/api/sessions/load":
            name = params.get("name", "").strip()
            name = re.sub(r"[^\w\-_]", "", name)
            if not global_agent:
                self.send_json({"error": "Agent not initialized"}, status=400)
                return
            try:
                s_dir = _get_session_dir()
                filepath = s_dir / f"{name}.json"
                if not filepath.exists():
                    self.send_json({"error": "Session not found"}, status=400)
                    return
                content = filepath.read_text(encoding="utf-8")
                global_agent.history = _deserialize_history(content, global_agent._types)
                self.send_json({"status": "ok"})
            except Exception as e:
                self.send_json({"error": str(e)}, status=500)
                
        elif self.path == "/api/clear":
            if global_agent:
                global_agent.history.clear()
            self.send_json({"status": "ok"})
            
        else:
            self.send_error(404, "Not Found")

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def run_gui_server(agent: Ctx0anAgent, port: int = 8080) -> None:
    global global_agent, GUI_MODE
    global_agent = agent
    GUI_MODE = True
    
    server = ThreadingHTTPServer(("127.0.0.1", port), Ctx0anHTTPRequestHandler)
    url = f"http://127.0.0.1:{port}"
    
    ui.success(f"Starting Ctx0an Web GUI on {url} ...")
    ui.info("Press Ctrl+C in this terminal to stop the server.")
    
    def open_browser():
        time.sleep(1.0)
        try:
            webbrowser.open(url)
        except Exception:
            ui.info(f"Could not open browser automatically. Please open {url} manually.")
        
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        ui.info("\nStopping Ctx0an Web GUI...")
    finally:
        server.server_close()
        ui.success("Server stopped.")


def interactive_loop(agent: Ctx0anAgent) -> None:
    """Interactive workspace mode: multi-turn session with memory (TUI dashboard)."""
    import shutil
    import random
    import time

    def _center(text: str, width: int) -> str:
        clean_text = re.sub(r"\033\[[0-9;]*[mK]", "", text)
        padding = max(0, (width - len(clean_text)) // 2)
        return " " * padding + text

    def draw_tui_logo(width: int) -> None:
        logo = [
            r"  ___  ____  _  _   __     __   _  _ ",
            r" / __)(_  _)( \/ ) /  \   / _\ ( \( )",
            r"( (__   )(   )  ( (  O ) /    \ )  ( ",
            r" \___) (__) (_/\_) \__/  \_/\_/(_\_) "
        ]
        
        # Clear screen
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()
        print("\n")
        
        symbols = "@#%&*+=-::"
        
        # Glitch decrypter animation (4 steps)
        for stage in range(4):
            lines = []
            for line in logo:
                animated_chars = []
                for char in line:
                    if char == " ":
                        animated_chars.append(" ")
                    else:
                        if stage == 3:
                            animated_chars.append(char)
                        elif stage == 2 and random.random() > 0.2:
                            animated_chars.append(char)
                        elif stage == 1 and random.random() > 0.6:
                            animated_chars.append(char)
                        else:
                            animated_chars.append(random.choice(symbols))
                lines.append("".join(animated_chars))
                
            if stage == 0:
                color = _ANSI.RED
            elif stage == 1:
                color = _ANSI.YELLOW
            elif stage == 2:
                color = _ANSI.BLUE
            else:
                color = _ANSI.CYAN
                
            for line in lines:
                print(_center(_ansi(line, _ANSI.BOLD, color), width))
                
            if stage < 3:
                time.sleep(0.08)
                sys.stdout.write("\033[4A\r")
                sys.stdout.flush()
                
        print(_center(_ansi(f"v{__version__}", _ANSI.DIM), width))
        print("\n")

    # Initial boot screen animation
    width = shutil.get_terminal_size().columns
    draw_tui_logo(width)

    commands = [
        ("/help", "show help"),
        ("/sessions", "list sessions"),
        ("/new", "start a new session"),
        ("/model", "switch model"),
        ("/staged", "list staged files"),
        ("/exit", "exit the app")
    ]
    for cmd, desc in commands:
        padding = 12 - len(cmd)
        line_text = f"{_ansi(cmd, _ANSI.BOLD, _ANSI.CYAN)}{' ' * padding}{_ansi(desc, _ANSI.DIM)}"
        print(_center(line_text, width))
    print("\n" * 3)

    while True:
        try:
            width = shutil.get_terminal_size().columns

            # Draw input box line
            divider = _ansi("─" * width, _ANSI.DIM)
            print(divider)

            # Draw footer line
            left_footer = _ansi("enter", _ANSI.BOLD) + _ansi(" send", _ANSI.DIM)
            model_display = MODEL_DESCRIPTIONS.get(agent.model_name, agent.model_name)
            right_footer = _ansi(model_display, _ANSI.DIM)

            clean_left = "enter send"
            spacing = width - len(clean_left) - len(model_display)
            footer = left_footer + (" " * max(0, spacing)) + right_footer
            print(footer)

            # Move cursor up 2 lines to the input line, carriage return, clear line
            sys.stdout.write("\033[2A\r\033[K")
            sys.stdout.write(_ansi(" > ", _ANSI.BOLD, _ANSI.CYAN))
            sys.stdout.flush()

            user_text = input()
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n\n")
            sys.stdout.flush()
            break

        command = user_text.strip()
        if not command:
            # Move down 2 lines to restore normal flow
            sys.stdout.write("\033[2B\r")
            sys.stdout.flush()
            continue

        # Overwrite the footer line and divider line before running command
        sys.stdout.write("\033[1B\r\033[K\033[1A\r")
        sys.stdout.flush()

        low = command.lower()
        if low in ("exit", "quit", ":q", "/exit", "/quit"):
            break
        if low == "/help":
            ui.echo(HELP_TEXT)
            continue
        if low == "/status":
            ui.info(
                f"current model: {agent.model_name} | "
                f"history items: {len(agent.history)} | "
                f"staged files: {len(agent.staged_files)}"
            )
            continue
        if low in ("/clear", "/new"):
            agent.history.clear()
            draw_tui_logo(width)
            for cmd, desc in commands:
                padding = 12 - len(cmd)
                line_text = f"{_ansi(cmd, _ANSI.BOLD, _ANSI.CYAN)}{' ' * padding}{_ansi(desc, _ANSI.DIM)}"
                print(_center(line_text, width))
            print("\n" * 3)
            ui.success("Conversation cleared / New session started.")
            continue
        if low == "/model":
            ui.info(f"current model: {agent.model_name}")
            continue
        if low.startswith("/model "):
            parts = command.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                ui.info(f"current model: {agent.model_name}")
                continue
            agent.model_name = resolve_model(parts[1])
            ui.success(f"Model switched to {agent.model_name}")
            continue

        # File staging commands
        if low.startswith("/add "):
            parts = command.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                ui.error("Usage: /add <file_path>")
                continue
            f_path = parts[1].strip()
            try:
                p = _resolve_path(f_path)
                if not p.exists():
                    ui.error(f"File not found: {f_path}")
                elif p.is_dir():
                    ui.error(f"'{f_path}' is a directory. You can only stage files.")
                else:
                    agent.staged_files.add(str(p))
                    ui.success(f"Staged file: {p.name} ({os.path.relpath(p)})")
            except Exception as e:
                ui.error(f"Error staging file: {e}")
            continue

        if low.startswith("/drop "):
            parts = command.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                ui.error("Usage: /drop <file_path>")
                continue
            f_path = parts[1].strip()
            try:
                p = _resolve_path(f_path)
                target_str = str(p)
                found = False
                if target_str in agent.staged_files:
                    agent.staged_files.remove(target_str)
                    found = True
                else:
                    for path_str in list(agent.staged_files):
                        if Path(path_str).name == f_path or os.path.relpath(path_str) == f_path:
                            agent.staged_files.remove(path_str)
                            found = True
                            p = Path(path_str)
                            break
                if found:
                    ui.success(f"Dropped staged file: {p.name}")
                else:
                    ui.error(f"File not staged: {f_path}")
            except Exception as e:
                ui.error(f"Error dropping file: {e}")
            continue

        if low == "/staged":
            if not agent.staged_files:
                ui.info("No files currently staged.")
            else:
                ui.info("Staged files (pinned context):")
                for path_str in sorted(agent.staged_files):
                    p = Path(path_str)
                    ui.echo(f"  {p.name} ({os.path.relpath(p)})")
            continue

        # Session serialization commands
        if low.startswith("/save "):
            parts = command.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                ui.error("Usage: /save <session_name>")
                continue
            name = parts[1].strip()
            name = re.sub(r"[^\w\-_]", "", name)
            if not name:
                ui.error("Invalid session name.")
                continue
            try:
                s_dir = _get_session_dir()
                filepath = s_dir / f"{name}.json"
                serialized = _serialize_history(agent.history, agent._types)
                filepath.write_text(serialized, encoding="utf-8")
                ui.success(f"Session saved to {filepath}")
            except Exception as e:
                ui.error(f"Failed to save session: {e}")
            continue

        if low == "/sessions":
            try:
                s_dir = _get_session_dir()
                files = sorted(s_dir.glob("*.json"))
                if not files:
                    ui.info("No saved sessions found.")
                else:
                    ui.info("Saved sessions:")
                    for f in files:
                        ui.echo(f"  {f.stem} (modified: {time.ctime(f.stat().st_mtime)})")
            except Exception as e:
                ui.error(f"Failed to list sessions: {e}")
            continue

        if low.startswith("/load "):
            parts = command.split(None, 1)
            if len(parts) < 2 or not parts[1].strip():
                ui.error("Usage: /load <session_name>")
                continue
            name = parts[1].strip()
            name = re.sub(r"[^\w\-_]", "", name)
            try:
                s_dir = _get_session_dir()
                filepath = s_dir / f"{name}.json"
                if not filepath.exists():
                    ui.error(f"Session '{name}' not found. Use /sessions to list available sessions.")
                    continue
                content = filepath.read_text(encoding="utf-8")
                agent.history = _deserialize_history(content, agent._types)
                ui.success(f"Session '{name}' loaded with {len(agent.history)} turns.")
            except Exception as e:
                ui.error(f"Failed to load session: {e}")
            continue

        try:
            agent.run_turn(user_text)
        except KeyboardInterrupt:
            ui.warn("turn interrupted by user.")
        except Exception as exc:
            hint = ""
            if (getattr(exc, "code", None)
                    or getattr(exc, "status_code", None)) == 429:
                hint = " Quota exhausted — try '/model flash'."
            ui.error(f"API error: {exc}.{hint}")
    ui.info("Session ended. Bye.")


def build_parser() -> argparse.ArgumentParser:
    epilog = """examples:
    ctx0an.py                                interactive workspace mode
    ctx0an.py --gui                          launch the interactive Web Chat UI
    ctx0an.py "create a flask app and run it on port 8080"
    ctx0an.py -m flash "fix the traceback in main.py and rerun it"
    ctx0an.py write a python script to parse logs and run it
"""
    parser = argparse.ArgumentParser(
        prog="ctx0an",
        description="Ctx0an — an autonomous software-development agent "
                    "for Termux, powered by Gemini.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="Task to execute. Omit for interactive mode. The words are "
             "joined, so quoting is optional.",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help="Model to use: 'pro' (gemini-2.5-pro, default), 'flash' "
             "(gemini-2.5-flash), or any Gemini model id.",
    )
    parser.add_argument(
        "-g", "--gui",
        action="store_true",
        help="Launch the self-contained Web Chat UI.",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8080,
        help="Port to run the Web GUI server on (default: 8080).",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def setup_missing_api_key(env_var: str, provider: str, ref_url: str) -> str:
    """Prompt the user for a missing API key and save it to their shell configuration."""
    if not sys.stdin.isatty() or GUI_MODE:
        return ""
    
    print(_ansi(f"\n[!] Configuration missing: {env_var} is not set.", _ANSI.YELLOW, _ANSI.BOLD))
    print(f"You can get an API key from: {ref_url}")
    try:
        response = input("Would you like to configure it now? [y/N]: ").strip().lower()
        if response not in ('y', 'yes'):
            return ""
        
        key = input(f"Enter your {env_var}: ").strip()
        if not key:
            return ""
        
        if os.name == 'nt':
            try:
                subprocess.run(["setx", env_var, key], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(_ansi(f"[+] Successfully set environment variable {env_var} permanently via setx.", _ANSI.GREEN))
                return key
            except Exception as e:
                print(_ansi(f"Could not set environment variable via setx: {e}", _ANSI.RED))
                return ""
        else:
            # Detect shell config file
            home = Path.home()
            candidates = [home / ".bashrc", home / ".zshrc", home / ".bash_profile", home / ".profile"]
            config_file = None
            for cand in candidates:
                if cand.exists():
                    config_file = cand
                    break
            if not config_file:
                config_file = home / ".bashrc"
                
            try:
                content = ""
                if config_file.exists():
                    content = config_file.read_text(encoding="utf-8")
                
                # Check if already present in some format
                if f"export {env_var}=" not in content:
                    prefix = "\n" if content and not content.endswith("\n") else ""
                    with open(config_file, "a", encoding="utf-8") as f:
                        f.write(f"{prefix}export {env_var}='{key}'\n")
                    print(_ansi(f"[+] Successfully saved API key to {config_file}", _ANSI.GREEN))
                    print(_ansi("Please reload your shell (e.g. source ~/.bashrc) for changes to take effect in other sessions.", _ANSI.DIM))
                else:
                    print(_ansi(f"Note: {env_var} is already referenced in {config_file}. Not modifying it.", _ANSI.YELLOW))
                return key
            except Exception as e:
                print(_ansi(f"Could not save key to file: {e}", _ANSI.RED))
                return ""
    except (KeyboardInterrupt, EOFError):
        print()
        return ""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = resolve_model(args.model)

    custom_cfg = CUSTOM_MODELS.get(model)
    if custom_cfg:
        provider = custom_cfg.get("provider", "openai")
        default_env = "GEMINI_API_KEY" if provider == "gemini" else ("CLAUDE_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY")
        env_var = custom_cfg.get("api_key_env", default_env)
        ref_url = custom_cfg.get("base_url", "https://platform.openai.com/")
    else:
        provider = MODEL_PROVIDERS.get(model, "gemini")
        if provider == "gemini":
            env_var = "GEMINI_API_KEY"
            ref_url = "https://aistudio.google.com/apikey"
        elif provider == "anthropic":
            env_var = "CLAUDE_API_KEY"
            ref_url = "https://console.anthropic.com/"
        elif provider == "openai":
            env_var = "OPENAI_API_KEY"
            ref_url = "https://platform.openai.com/api-keys"
        else:
            env_var = ""
            ref_url = ""

    if env_var == "CLAUDE_API_KEY":
        api_key = (os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")).strip()
    elif env_var:
        api_key = os.environ.get(env_var, "").strip()
    else:
        api_key = ""

    if not api_key:
        api_key = setup_missing_api_key(env_var, provider, ref_url)

    if not api_key:
        ui.error(
            f"{env_var} is not set for provider '{provider}'.\n"
            f"Get a key at {ref_url}, then:\n"
            f"    export {env_var}='your-key-here'\n"
            f"(add that line to ~/.bashrc or ~/.profile to persist it)"
        )
        return 1

    try:
        agent = Ctx0anAgent(model, api_key)
    except SystemExit:
        raise                               # missing SDK -> guided exit
    except Exception as exc:
        ui.error(f"failed to initialize the Gemini client: "
                 f"{type(exc).__name__}: {exc}")
        return 1

    try:
        if args.gui:
            run_gui_server(agent, args.port)
            return 0

        task = " ".join(args.task).strip()
        if task:
            ui.banner(model, mode="single-task")
            agent.run_turn(task)
        else:
            interactive_loop(agent)
    except KeyboardInterrupt:
        ui.warn("interrupted.")
        return 130
    except Exception as exc:
        hint = ""
        if (getattr(exc, "code", None)
                or getattr(exc, "status_code", None)) == 429:
            hint = " Quota exhausted — retry with '-m flash'."
        ui.error(f"fatal: {type(exc).__name__}: {exc}.{hint}")
        return 1
    finally:
        if hasattr(agent, "mcp_manager"):
            agent.mcp_manager.close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())