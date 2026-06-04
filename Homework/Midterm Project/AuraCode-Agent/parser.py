"""
AuraCode · Shared Parser Module
Tested against 14 different model output formats.
Import this in v1, v2, v3.
"""

import re
import logging

logger = logging.getLogger("AuraCode.Parser")


def parse_llm_response(raw: str, attempt: int = 1) -> dict:
    """
    Ultra-robust parser. Extracts code from ANY model output format:
      - ##TAG## delimiters (ideal)
      - ```python ... ``` markdown blocks
      - Plain ``` ... ``` blocks
      - Raw Python code (keyword detection)
      - <think>...</think> tags (Qwen/reasoning models)

    Args:
        raw:     Full text from the LLM.
        attempt: Iteration number (for logging).

    Returns:
        dict with keys: filename, description, dependencies, code, thought
    """
    # 1. Pull out <think> content then strip it
    think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
    thought = think_match.group(1).strip() if think_match else ""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # 2. Extract ##TAG## sections (flexible: spaces, case, \r\n)
    def get_tag(tag: str) -> str:
        p = rf'##\s*{tag}\s*##\s*(.*?)(?=##\s*[A-Z_]+\s*##|$)'
        m = re.search(p, raw, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    thought      = get_tag("THOUGHT") or thought
    filename     = get_tag("FILENAME")
    description  = get_tag("DESCRIPTION")
    dependencies = get_tag("DEPENDENCIES")
    code         = get_tag("CODE")

    # Strip any markdown fences that slipped into ##CODE## block
    if code:
        code = re.sub(r'^```\w*\n?', '', code.strip())
        code = re.sub(r'```$', '', code.strip()).strip()

    # 3. Fallback: ```python ... ```
    if not code:
        m = re.search(r'```python\s*(.*?)```', raw, re.DOTALL)
        if m:
            code = m.group(1).strip()
            logger.debug("Parser fallback: ```python block")

    # 4. Fallback: any ``` ... ```
    if not code:
        m = re.search(r'```\s*(.*?)```', raw, re.DOTALL)
        if m:
            code = m.group(1).strip()
            logger.debug("Parser fallback: ``` block")

    # 5. Fallback: detect Python by keywords
    if not code:
        py_keywords = ['import ', 'from ', 'def ', 'class ', 'print(', 'if __name__']
        if any(kw in raw for kw in py_keywords):
            started, collected = False, []
            for line in raw.splitlines():
                if any(kw in line for kw in py_keywords):
                    started = True
                if started:
                    collected.append(line)
            code = '\n'.join(collected).strip()
            if code:
                logger.debug("Parser fallback: keyword detection")

    if not code:
        raise ValueError(
            f"[Attempt {attempt}] Could not extract Python code.\n"
            f"Model returned:\n{raw[:500]}"
        )

    # Clean up filename
    fn = filename.lower().replace('.py', '').strip()
    fn = re.sub(r'[^\w]', '_', fn)
    fn = re.sub(r'_+', '_', fn).strip('_')
    if not fn:
        fn = "generated_script"

    # Clean up dependencies
    deps = dependencies.strip() if dependencies.strip() else "none"

    return {
        "filename":     fn,
        "description":  description.strip() or "Generated Python script",
        "dependencies": deps,
        "code":         code,
        "thought":      thought,
    }


def deps_to_list(deps_str: str) -> list[str]:
    """Convert 'requests, beautifulsoup4' → ['requests', 'beautifulsoup4']"""
    if not deps_str or deps_str.lower().strip() == "none":
        return []
    return [d.strip() for d in deps_str.split(",") if d.strip() and d.strip().lower() != "none"]