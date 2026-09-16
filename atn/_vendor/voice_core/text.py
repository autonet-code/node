"""Text cleanup for TTS, plus tool summary/narration helpers.

``strip_markdown`` is kevin's version with autonet's emoji-stripping fix grafted
in, so TTS no longer reads emoji names aloud (e.g. "microphone" for 🎤).
"""
from __future__ import annotations

import json
import os
import re

# Emoji / pictograph ranges — stripped so TTS doesn't read them as words.
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"     # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"     # transport & map symbols
    "\U0001F1E0-\U0001F1FF"     # flags
    "\U0001F900-\U0001F9FF"     # supplemental symbols
    "\U0001FA00-\U0001FA6F"     # chess symbols, extended-A
    "\U0001FA70-\U0001FAFF"     # symbols extended-A continued
    "\U00002702-\U000027B0"     # dingbats
    "\U0000FE00-\U0000FE0F"     # variation selectors
    "\U0000200D"                # zero-width joiner
    "\U000023F0-\U000023FA"     # misc technical
    "\U00002600-\U000026FF"     # misc symbols
    "\U0000203C\U00002049"      # exclamation marks
    "\U000020E3"                # combining enclosing keycap
    "\U00002934\U00002935"      # arrows
    "\U000025AA-\U000025FE"     # geometric shapes
    "]+"
)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting and emojis so TTS reads natural prose."""
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'^[\s\-:]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Strip emojis so TTS doesn't read them as words
    text = _EMOJI_RE.sub('', text)
    # Collapse Windows and Unix paths to basename
    text = re.sub(r'[A-Z]:\\(?:[^\\\s]+\\)*([^\\\s]+)', r'\1', text)
    text = re.sub(r'(?<!\w)/(?:[^/\s]+/)+([^/\s]+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def basename(path: str) -> str:
    name = os.path.basename(path)
    root, ext = os.path.splitext(name)
    return root if ext else name


def tool_summary(name: str, inp: dict) -> str:
    if name == "Bash":
        return inp.get("command", "")[:80]
    if name == "Read":
        return inp.get("file_path", "")
    if name in ("Edit", "Write"):
        return inp.get("file_path", "")
    if name == "Grep":
        return inp.get("pattern", "")
    if name == "Glob":
        return inp.get("pattern", "")
    if name == "Task":
        return inp.get("description", "")[:60]
    if name == "WebSearch":
        return inp.get("query", "")[:60]
    if name == "WebFetch":
        return inp.get("url", "")[:60]
    if name == "TodoWrite":
        return f"{len(inp.get('todos', []))} items"
    return json.dumps(inp)[:60]


def tool_narration(name: str, inp: dict) -> str:
    if name == "Read":
        return f"Reading {basename(inp.get('file_path', 'file'))}"
    if name == "Edit":
        return f"Editing {basename(inp.get('file_path', 'file'))}"
    if name == "Write":
        return f"Writing {basename(inp.get('file_path', 'file'))}"
    if name == "Bash":
        cmd = inp.get("command", "")
        first = cmd.split()[0] if cmd.split() else "command"
        first = os.path.basename(first)
        return f"Running {first}"
    if name == "Grep":
        return f"Searching for {inp.get('pattern', '')[:40]}"
    if name == "Glob":
        return f"Finding files matching {inp.get('pattern', '')[:40]}"
    if name == "Task":
        return f"Launching agent: {inp.get('description', 'task')[:50]}"
    if name == "WebSearch":
        return f"Searching the web for {inp.get('query', '')[:50]}"
    if name == "WebFetch":
        return "Fetching a web page"
    if name == "TodoWrite":
        return f"Updating task list, {len(inp.get('todos', []))} items"
    if name == "NotebookEdit":
        return "Editing notebook"
    return f"Using {name}"
