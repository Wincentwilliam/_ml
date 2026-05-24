"""
generate.py
===========
Second-Order Markov Chain Model for Text Generation
----------------------------------------------------
Author  : Claude (Anthropic)
Purpose : Learns the statistical structure of a text corpus (tw.txt) and
          generates new text that stylistically mimics the original.

ML Concept — The Markov Property (2nd Order)
---------------------------------------------
A 1st-order Markov Chain predicts the next state using only the current state.
A 2nd-order Markov Chain extends this by conditioning on the TWO most recent
states:

    P(w_t | w_{t-1}, w_{t-2}, ..., w_1)  ≈  P(w_t | w_{t-1}, w_{t-2})

This "limited memory" assumption trades theoretical exactness for tractability:
we can estimate the conditional probabilities directly from observed bigram
(pair) frequencies in the training corpus, with no gradient descent required.

Transition Table (the "model"):
    key   → (word_{n-2}, word_{n-1})   # a 2-word context window
    value → [word_n, word_n, word_n, ...]  # every observed successor
                                            # (duplicates encode frequency)

At generation time we call random.choice(value), which implicitly samples
from the empirical unigram distribution of successors for that context.
"""

import re
import random
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_text(filepath: str = "tw.txt") -> str:
    """
    Read the raw text corpus from disk.

    Parameters
    ----------
    filepath : str
        Path to the input text file (default: 'tw.txt').

    Returns
    -------
    str
        The full raw text as a Unicode string.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"[ERROR] Training corpus '{filepath}' not found.\n"
            "Please place your text file in the same directory as generate.py "
            "and name it 'tw.txt' (or pass a custom path)."
        )
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    print(f"[INFO] Loaded '{filepath}' — {len(text):,} characters.")
    return text


# ---------------------------------------------------------------------------
# 2. Preprocessing / Tokenisation
# ---------------------------------------------------------------------------

def tokenize(text: str, mode: str = "auto") -> list[str]:
    """
    Convert raw text into a flat list of tokens.

    Two modes are supported:

    * ``"word"``  — split on whitespace / punctuation boundaries.
                    Best for space-delimited languages (English, etc.).
    * ``"char"``  — split into individual characters (including Chinese/CJK).
                    Best for character-based languages (Traditional/Simplified
                    Chinese, Japanese kanji, etc.).
    * ``"auto"``  — detect automatically: if more than 20 % of non-whitespace
                    characters are CJK codepoints, use char mode; otherwise
                    use word mode.

    Parameters
    ----------
    text : str
        Raw input text.
    mode : str
        Tokenisation strategy: ``"auto"`` | ``"word"`` | ``"char"``.

    Returns
    -------
    list[str]
        Ordered list of tokens ready for Markov training.
    """
    # --- auto-detect language ---
    if mode == "auto":
        non_ws = [c for c in text if not c.isspace()]
        if not non_ws:
            return []
        cjk_count = sum(1 for c in non_ws if "\u4e00" <= c <= "\u9fff"
                        or "\u3400" <= c <= "\u4dbf"
                        or "\uf900" <= c <= "\ufaff")
        mode = "char" if (cjk_count / len(non_ws)) > 0.20 else "word"
        print(f"[INFO] Auto-detected tokenisation mode: '{mode}' "
              f"(CJK ratio = {cjk_count / len(non_ws):.1%})")

    if mode == "char":
        # Keep every non-whitespace character as its own token.
        tokens = [c for c in text if not c.isspace()]
    else:  # word mode
        # Normalise whitespace, then split on whitespace.
        # Optionally strip leading/trailing punctuation from each token
        # so "hello," and "hello" map to the same token.
        text = re.sub(r"\s+", " ", text).strip()
        raw_tokens = text.split()
        tokens = [t.strip("\"'「」『』【】()（）[]《》<>") for t in raw_tokens]
        tokens = [t for t in tokens if t]  # drop empty strings

    print(f"[INFO] Corpus tokenised → {len(tokens):,} tokens.")
    return tokens


# ---------------------------------------------------------------------------
# 3. Model Training — Build the Transition Table
# ---------------------------------------------------------------------------

def train(tokens: list[str]) -> dict[tuple[str, str], list[str]]:
    """
    Build the 2nd-order Markov transition table from the token sequence.

    Algorithm
    ---------
    Slide a window of width 3 across the token list:

        index :  0    1    2    3    4  …
        token :  A    B    C    D    E  …
                 ↑────↑                   ← key   = (A, B)
                      ↑────↑              ← successor stored: C
                      ↑────↑              ← key   = (B, C)
                           ↑────↑         ← successor stored: D
                           …

    Because we append *every* occurrence of a successor (not a unique set),
    the resulting list implicitly encodes frequency.  Calling
    ``random.choice(transitions[(A, B)])`` samples proportionally to
    the empirical conditional probability  P(next | A, B).

    Parameters
    ----------
    tokens : list[str]
        Ordered list of tokens from the corpus.

    Returns
    -------
    dict[tuple[str, str], list[str]]
        Mapping  (word_{n-2}, word_{n-1})  →  [possible next words …]
    """
    if len(tokens) < 3:
        raise ValueError(
            "[ERROR] Corpus is too small — need at least 3 tokens to train "
            "a 2nd-order Markov model."
        )

    # defaultdict(list) avoids explicit key-existence checks.
    transitions: dict[tuple[str, str], list[str]] = defaultdict(list)

    for i in range(len(tokens) - 2):
        # The 2-token context window (the "state" in ML terms)
        context: tuple[str, str] = (tokens[i], tokens[i + 1])
        # The word the model should learn to predict from this context
        successor: str = tokens[i + 2]
        transitions[context].append(successor)

    print(f"[INFO] Training complete — {len(transitions):,} unique bigram "
          "contexts learned.")
    return transitions


# ---------------------------------------------------------------------------
# 4. Text Generation
# ---------------------------------------------------------------------------

def generate(
    transitions: dict[tuple[str, str], list[str]],
    length: int = 100,
    seed: tuple[str, str] | None = None,
    mode: str = "word",
) -> str:
    """
    Generate a new token sequence using the learned Markov transitions.

    Generation Loop
    ---------------
    1. Choose a starting bigram (``seed``).  If none is supplied, pick one
       at random from the keys of the transition table — guaranteeing the
       seed has at least one known successor.
    2. At each step, look up ``transitions[(prev_prev, prev)]``.
    3. Call ``random.choice(...)`` to sample the next token proportionally
       to its observed frequency (the ML "inference" step).
    4. Slide the window: ``prev_prev ← prev``, ``prev ← next_token``.
    5. Repeat until ``length`` tokens have been generated, *or* a dead end
       is reached (the current bigram was never seen during training).

    Dead-End Handling
    -----------------
    When the current bigram has no recorded successors (rare but possible if
    the corpus ends mid-sentence or if ``seed`` was manually supplied),
    the function randomly restarts from a known context rather than crashing.

    Parameters
    ----------
    transitions : dict
        The trained transition table from :func:`train`.
    length : int
        Number of tokens to generate (default: 100).
    seed : tuple[str, str] or None
        Optional starting bigram ``(word_0, word_1)``.  If ``None``, a
        random key from ``transitions`` is chosen.
    mode : str
        ``"word"`` joins tokens with spaces; ``"char"`` joins without spaces.

    Returns
    -------
    str
        The generated text as a single string.
    """
    if not transitions:
        raise ValueError("[ERROR] Transition table is empty — train the model first.")

    all_seeds = list(transitions.keys())

    # --- choose starting bigram ---
    if seed is None:
        current = random.choice(all_seeds)
    else:
        if seed not in transitions:
            print(f"[WARN] Seed {seed!r} not found in transition table; "
                  "choosing a random seed instead.")
            current = random.choice(all_seeds)
        else:
            current = seed

    output_tokens: list[str] = list(current)  # start with the seed tokens

    # --- generation loop ---
    for step in range(length - 2):  # -2 because seed already contributes 2 tokens
        prev_prev, prev = output_tokens[-2], output_tokens[-1]
        context = (prev_prev, prev)

        if context not in transitions:
            # Dead end — restart from a random known context
            print(f"[INFO] Dead end at step {step + 1}; restarting from a "
                  "random seed.")
            current = random.choice(all_seeds)
            output_tokens.extend(list(current))
            continue

        # Sample the next token from the empirical distribution
        next_token: str = random.choice(transitions[context])
        output_tokens.append(next_token)

    separator = "" if mode == "char" else " "
    return separator.join(output_tokens)


# ---------------------------------------------------------------------------
# 5. Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full pipeline:
        load → tokenize → train → generate → print
    """
    # --- configurable parameters ---
    CORPUS_FILE   = "tw.txt"
    GENERATE_LEN  = 150       # number of tokens to generate
    TOKEN_MODE    = "auto"    # "auto" | "word" | "char"
    RANDOM_SEED   = None      # None = different output every run

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        print(f"[INFO] Random seed fixed at {RANDOM_SEED} (reproducible run).")

    # Step 1 — Load
    try:
        raw_text = load_text(CORPUS_FILE)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    # Step 2 — Tokenise
    tokens = tokenize(raw_text, mode=TOKEN_MODE)
    if not tokens:
        print("[ERROR] No tokens found after preprocessing. Check your input file.")
        sys.exit(1)

    # Determine the effective mode after auto-detection for the join step
    effective_mode = TOKEN_MODE
    if TOKEN_MODE == "auto":
        non_ws = [c for c in raw_text if not c.isspace()]
        cjk_count = sum(1 for c in non_ws if "\u4e00" <= c <= "\u9fff")
        effective_mode = "char" if non_ws and (cjk_count / len(non_ws)) > 0.20 else "word"

    # Step 3 — Train
    try:
        transitions = train(tokens)
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    # Step 4 — Generate
    print(f"\n[INFO] Generating {GENERATE_LEN} tokens …\n")
    try:
        output = generate(
            transitions,
            length=GENERATE_LEN,
            seed=None,          # random start
            mode=effective_mode,
        )
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    # Step 5 — Display
    print("=" * 60)
    print("GENERATED TEXT")
    print("=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()