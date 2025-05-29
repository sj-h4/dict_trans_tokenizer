# Dict Trans Tokenizer

A Python package for cross-lingual tokenizer alignment using bilingual dictionaries and recurrent alignment algorithms.

## Overview

Dict Trans Tokenizer enables you to:

- Train BPE tokenizers from bilingual dictionary data
- Create aligned corpora from bilingual dictionaries
- Perform recurrent token alignment between source and target tokenizers
- Remap pre-trained models to use aligned tokenizers

This is particularly useful for adapting pre-trained language models to new languages or domains using bilingual dictionaries.

## Installation

```bash
uv add dict-trans-tokenizer git+https://github.com/sj-h4/dict_trans_tokenizer.git
```

### Dependencies

- **fast_align**: Required for token alignment

> [!NOTE]
> Please use <https://github.com/FremyCompany/fast_align.git>, which is used in [TransTokenizer](https://openreview.net/forum?id=sBxvoDhvao#discussion).

## Quick Start

See also the `example/basic_usage.py` for a complete example.

### Basic Usage

```python
from dict_trans_tokenizer import (
    run_recurrent_alignment,
    train_bpe_tokenizer,
    load_bilingual_dict,
)

# Train tokenizers and run alignment
run_recurrent_alignment(
    source_model="roberta-base",  # or path to source model
    target_tokenizer="path/to/target/tokenizer",
    dictionary="path/to/bilingual_dict.json",
    corpus_path="output/corpus.moses",
    fast_align_path="fast_align",
    alignment_mode="token",  # or "word"
    mapping_mode="replace",
    min_count=10,
    output_dir="output/",
    logging_level="INFO",
)
```

### Step-by-Step Workflow

```python
import tempfile
from pathlib import Path
from dict_trans_tokenizer import (
    train_bpe_tokenizer,
    create_aligned_corpus,
    run_recurrent_alignment,
    load_bilingual_dict,
    AlignmentMode,
)

# 1. Load bilingual dictionary
dictionary = load_bilingual_dict("path/to/dict.json")

# 2. Train target language tokenizer
target_words = [entry.entry.lower().strip() for entry in dictionary]
train_bpe_tokenizer(
    train_words=target_words,
    output_dir="target_tokenizer/",
    vocab_size=1000,
    show_progress=True
)

# 3. Create aligned corpus
corpus_path = create_aligned_corpus(
    source_tokenizer=source_tokenizer,
    target_tokenizer=target_tokenizer,
    dictionary=dictionary,
    output_path="aligned_corpus.moses",
    alignment_mode=AlignmentMode.TOKEN,
)

# 4. Run recurrent alignment
run_recurrent_alignment(
    source_model="roberta-base",
    target_tokenizer="target_tokenizer/tokenizer-fast",
    dictionary="path/to/dict.json",
    corpus_path="corpus.moses",
    fast_align_path="fast_align",
    alignment_mode="token",
    mapping_mode="replace",
    min_count=5,
    output_dir="aligned_model/",
)
```

## Dictionary Format

The bilingual dictionary should be a JSON file with the following structure:

```json
[
  {
    "entry": "hello",
    "definitions": ["greeting", "hi", "welcome"]
  },
  {
    "entry": "world",
    "definitions": ["earth", "planet", "globe"]
  }
]
```

Each entry contains:

- `entry`: The target language word
- `definitions`: List of source language translations/definitions

## API Reference

### Core Functions

#### `run_recurrent_alignment()`

Main function that performs the complete recurrent alignment process.

**Parameters:**

- `source_model` (str): Path to source model or HuggingFace model name
- `target_tokenizer` (str): Path to target tokenizer
- `dictionary` (str): Path to bilingual dictionary JSON file
- `corpus_path` (str): Path where aligned corpus will be saved
- `fast_align_path` (str): Path to fast_align binary
- `alignment_mode` (str): "token" or "word" alignment mode
- `mapping_mode` (str): "replace" (currently only supported mode)
- `min_count` (int): Minimum count threshold for token consideration
- `output_dir` (str): Directory to save results
- `logging_level` (str): Logging level (default: "INFO")
- `seed` (int): Random seed (default: 42)

#### `train_bpe_tokenizer()`

Train a BPE tokenizer from word list.

**Parameters:**

- `train_words` (list[str]): List of words to train on
- `output_dir` (str): Directory to save tokenizer
- `vocab_size` (int): Vocabulary size
- `show_progress` (bool): Whether to show training progress
- `special_tokens` (list[str]): Special tokens (optional)

#### `create_aligned_corpus()`

Create aligned corpus from bilingual dictionary.

**Parameters:**

- `source_tokenizer`: Source language tokenizer
- `target_tokenizer`: Target language tokenizer
- `dictionary`: List of bilingual dictionary entries
- `output_path` (str): Path to save aligned corpus
- `alignment_mode`: AlignmentMode.TOKEN or AlignmentMode.WORD

## Development

### Setup

```bash
uv run lefthook install
```

### Running Tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check --fix
```

## Citation

TBA
