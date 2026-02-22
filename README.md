# LLMTextPreprocessingFoundations

Notebook based on **Build a Large Language Model (From Scratch)** by Sebastian Raschka, Chapter 2.

## Contents

| File | Description |
|---|---|
| `embeddings.ipynb` | Main notebook with core code, explanations and experiment |
| `the-verdict.txt` | Short story used as the training corpus |

## What the notebook covers

- **Tokenization** — regex-based tokenizer + BPE via `tiktoken`
- **Vocabulary building** — string↔integer mappings and special tokens
- **Sliding-window dataset** — generating (input, target) pairs for next-token prediction
- **Token + position embeddings** — `nn.Embedding` layers and why both are needed
- **Personal explanations** — 4 markdown cells connecting each step to LLM and agentic system design
- **Experiment** — effect of `max_length` and `stride` on sample count

## Experiment highlights

With a ~4,690-token corpus:

| `max_length` | `stride` | Samples |
|:---:|:---:|:---:|
| 32 | 32 | 146 |
| 32 | 16 | 292 |
| 32 | 8 | 583 |
| 32 | 1 | 4,658 |

Setting `stride=1` yields **31× more samples** than no overlap — useful for small corpora.

## Requirements

```bash
pip install tiktoken torch jupyter
```

The notebook runs with graceful fallbacks if `torch` or `tiktoken` are not installed — all pure-Python sections still execute.

## Key concept

> Embeddings encode meaning not because they are programmed to, but because semantic similarity is what the model must compress to predict the next token correctly. Backpropagation pushes contextually similar tokens toward the same region of the embedding space — the distributional hypothesis implemented as gradient descent.
