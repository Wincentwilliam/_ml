<div align="center">

# 🔗 Second-Order Markov Chain Text Generator

### 用馬可夫鏈學會文字生成 — Learn Text Generation with Markov Chains

> **Built by: Wincent (Student Homework — Machine Learning Course)**

</div>

---

> **Transparency Notice:** This project was developed with the assistance of [Claude](https://claude.ai), Anthropic's AI assistant, to ensure code quality and educational clarity.

---

<div align="center">

![Python](https://img.shields.io/badge/Language-Python%203.10+-blue)
![NLP](https://img.shields.io/badge/Topic-Natural%20Language%20Processing-green)
![ML](https://img.shields.io/badge/Field-Machine%20Learning-orange)
![Level](https://img.shields.io/badge/Level-University%20Homework-purple)

*A clean, well-documented implementation of a 2nd-Order Markov Chain for text generation — no neural networks, no external libraries, just pure Python and probability.*

</div>

---

## Table of Contents

| Section | Description |
|:-------:|-------------|
| [What Is This?](#-what-is-this) | Project overview |
| [The ML Concept](#-the-ml-concept--markov-property) | Theory explained simply |
| [Project Structure](#-project-structure) | File layout |
| [Quick Start](#-quick-start) | How to run it |
| [How It Works](#-how-it-works-step-by-step) | Pipeline walkthrough |
| [Configuration](#-configuration) | Tunable parameters |
| [Example Output](#-example-output) | Sample generated text |
| [Limitations](#-limitations--next-steps) | Honest trade-offs |
| [Further Reading](#-further-reading) | Learn more |

---

## 🤔 What Is This?

This is a **Second-Order Markov Chain text generator** — a classic, foundational algorithm in Natural Language Processing (NLP). You give it a `.txt` file, it reads and "learns" the patterns of the text, and then it generates brand-new sentences that statistically mimic the original.

### What Makes It "2nd Order"?

| Model | Memory | Example |
|-------|--------|---------|
| 1st-Order | Looks back 1 word | `P(next \| "cat")` |
| **2nd-Order** | **Looks back 2 words** | **`P(next \| "the", "cat")`** |
| Nth-Order | Looks back N words | `P(next \| w₁, w₂, …, wₙ)` |

Using **2 words of context** produces noticeably more coherent output than a 1st-order model — it sounds more like a real sentence — while still being completely transparent and explainable, with no black-box magic involved.

---

## 🧠 The ML Concept — Markov Property

### The Big Idea

A **Markov Chain** makes one key assumption: *the future depends only on the recent past, not the entire history.* This is called the **Markov Property**, and it's what makes the model tractable — we can estimate probabilities directly from counted frequencies in the training text.

```
Full history model:  P(w_t | w_{t-1}, w_{t-2}, ..., w_1)   ← impossible to scale
2nd-Order Markov:    P(w_t | w_{t-1}, w_{t-2})              ← tractable!
```

### Training Phase — Build the Transition Table

The model slides a 3-token window across the entire corpus and records every successor it sees:

```
Corpus:  "the cat sat on the cat mat"

  ("the",  "cat") → ["sat", "mat"]   ← "sat" AND "mat" both followed this pair
  ("cat",  "sat") → ["on"]
  ("sat",  "on")  → ["the"]
  ("on",   "the") → ["cat"]
  ("cat",  "mat") → []               ← end of corpus, no successor
```

> 💡 **Key insight:** Duplicates are stored intentionally. If `"sat"` appears 3× and `"mat"` appears 1× after the same pair, the list becomes `["sat", "sat", "sat", "mat"]`. Calling `random.choice()` on this naturally samples `"sat"` 75% of the time — this *is* the probability distribution, stored structurally.

### Generation Phase — Inference Loop

```
Step 1 → Pick a random seed pair, e.g. ("the", "cat")
Step 2 → Look up transitions[("the", "cat")] → ["sat", "mat"]
Step 3 → Sample: random.choice(…) → "sat"
Step 4 → Slide window: new context = ("cat", "sat")
Step 5 → Repeat until desired length is reached
```

No matrix algebra. No backpropagation. Just counting and sampling. That's the beauty of it.

---

## 📁 Project Structure

```
Homework/06/
├── generate.py   ← main script (load → tokenise → train → generate)
├── tw.txt        ← training corpus (your text data goes here)
└── README.md     ← this file
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.10 or higher**
- **No third-party packages required** — uses Python standard library only (`random`, `re`, `collections`, `pathlib`)

### Step 1 — Add Your Corpus

Place a text file named `tw.txt` in the same folder as `generate.py`.

- **English text** → tokenised automatically by word
- **Chinese / CJK text** → auto-detected and tokenised character-by-character
- File must be saved as **UTF-8 encoding**

### Step 2 — Run

```bash
python generate.py
```

### Step 3 — See the Output

```
[INFO] Loaded 'tw.txt' — 7,753 characters.
[INFO] Auto-detected tokenisation mode: 'word' (CJK ratio = 0.0%)
[INFO] Corpus tokenised → 1,443 tokens.
[INFO] Training complete — 1,245 unique bigram contexts learned.
[INFO] Generating 150 tokens …

============================================================
GENERATED TEXT
============================================================
hey how are you doing today i am doing great thanks for asking
how about you yeah totally i get that life gets hectic sometimes
but you have to keep pushing forward absolutely you just have to
take it one day at a time ...
============================================================
```

---

## 🔧 How It Works — Step by Step

The script is split into five clean, single-responsibility functions:

| Function | What It Does |
|----------|-------------|
| `load_text(filepath)` | Opens `tw.txt` with UTF-8 encoding; raises a clear error if the file is missing |
| `tokenize(text, mode)` | Cleans whitespace; auto-detects CJK vs English; splits into word or character tokens |
| `train(tokens)` | Slides a 3-token window across the corpus to build the transition dictionary |
| `generate(transitions, length, seed, mode)` | Samples the learned distribution token-by-token; handles dead ends gracefully |
| `main()` | Wires all four steps together; all tunable parameters live here |

### The Compilation Pipeline

```
tw.txt (raw text)
       │
       ▼
 load_text()
 Reads file, UTF-8 safe
       │
       ▼
 tokenize()
 Cleans whitespace → splits into tokens
 ["hey", "how", "are", "you", ...]
       │
       ▼
 train()
 Builds transition table
 {("hey","how"): ["are"], ("how","are"): ["you","you","doing"], ...}
       │
       ▼
 generate()
 Samples the table token-by-token
 "hey how are you doing today ..."
       │
       ▼
 Printed to console ✅
```

---

## ⚙️ Configuration

All tunable parameters are at the top of `main()` in `generate.py`:

```python
CORPUS_FILE  = "tw.txt"   # path to your training text
GENERATE_LEN = 150        # number of tokens to generate
TOKEN_MODE   = "auto"     # "auto" | "word" | "char"
RANDOM_SEED  = None       # None = different output every run
                          # set to 42 (any integer) for reproducible output
```

### Tips

- **Longer output?** Increase `GENERATE_LEN` to `300` or `500`
- **Same output every run?** Set `RANDOM_SEED = 42`
- **Chinese text?** Set `TOKEN_MODE = "char"` or just leave it as `"auto"`
- **Better quality output?** Add more text to `tw.txt` — the more data, the richer the model

---

## 💡 Example Output

*Trained on conversational English (`tw.txt`):*

```
hey how are you doing today i am doing great thanks for asking how
about you yeah totally i get that life gets hectic sometimes but you
have to keep pushing forward absolutely you just have to take it one
day at a time and not stress too much that is really good advice i
try to do that but sometimes the stress just piles up i know what you
mean it can be really overwhelming but talking to someone always helps
```

*Trained on Chinese text (character mode):*

```
今天天氣很好，我們去公園散步，看見了很多小朋友在玩耍，天氣很好的
時候，大家都喜歡出來走走，公園裡的花開得很漂亮，讓人心情愉快…
```

---

## ⚠️ Limitations & Next Steps

This model is intentionally simple — it's a homework exercise, not a production system. Here are its honest trade-offs:

| Limitation | Why It Happens | How to Improve |
|-----------|---------------|----------------|
| Repetition loops | Small corpus with limited unique bigrams | Add more training data |
| No grammar understanding | Pure frequency counting, no syntax | Use RNN or Transformer |
| Dead ends possible | Some bigrams only appear at corpus end | Implement backoff to 1st-order |
| No punctuation | Stripped during tokenisation | Treat punctuation as tokens |
| Context window is fixed at 2 | Hardcoded 2nd-order | Generalise to N-th order |

### Where to Go From Here

```
Markov Chain (this project)
       │
       ▼
N-gram Language Model
       │
       ▼
Recurrent Neural Network (RNN / LSTM)
       │
       ▼
Transformer (Attention is All You Need)
       │
       ▼
GPT / Modern LLMs
```

Every step up this ladder adds more context, more parameters, and more understanding — but also more complexity and compute. This project is the foundation.

---

## 📚 Further Reading

- Jurafsky & Martin, *Speech and Language Processing* — Chapter 3: N-gram Language Models
- Manning & Schütze, *Foundations of Statistical NLP* — Chapter 6
- [Wikipedia: Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)
- [Wikipedia: N-gram Language Model](https://en.wikipedia.org/wiki/Language_model)

---

## Acknowledgments

- **Claude (Anthropic)** — For code structure, docstrings, and README drafting assistance
- **Professor** — For the assignment that made this possible
- **The Python Standard Library** — No pip install required 🎉

---

<div align="center">

**Happy Generating! 🎲**

*Last updated: May 2026*

</div>