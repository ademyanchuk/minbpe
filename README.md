# minbpe

A minimal, educational reimplementation of Byte Pair Encoding (BPE) – built as part of my learning journey through Andrej Karpathy's excellent [minbpe](https://github.com/karpathy/minbpe) repository and the corresponding [exercises.md](https://github.com/karpathy/minbpe/blob/master/exercise.md).

## 🎯 Goal
The primary goal of this repo is **not** to create the most efficient tokenizer, but to develop **deep understanding** of BPE through reimplementation, experimentation, and reflection. Every piece of code and every design decision here is a step toward becoming a better LLM research engineer by actively learning the internals of modern NLP systems. This repository intentionally implements only first 2 steps of above mentioned exercise, by focusing on the major concepts and leaving some details for the other time.

## 🛠 Features
- Pure Python implementation of BPE with zero dependencies
- Supports both basic and regex-based tokenization strategies
- Modular, easy to read and extend
- Built from first principles with step-by-step reasoning

## 📚 Inspired by
This work is directly inspired by:
- [Andrej Karpathy's minbpe repo](https://github.com/karpathy/minbpe)
- [exercises.md](https://github.com/karpathy/minbpe/blob/master/exercise.md) — particularly the tokenizer construction tasks

## 🔍 Key Insights from the Learning Journal

### 🧠 On Merge Order Consistency
> During encoding, merge order must match training. Merging `ab → X` before `aa → Z` may prevent `Z` from forming and break downstream merges like `Za → Y`. This disrupts consistency and can fragment tokens.

### 📉 Why Greedy Merge Order Matters
> Merge order affects compression and meaning. Out-of-order merges may create unexpected sequences that the model has never seen, hurting downstream performance.

### 🧪 Regex-Based Tokenization Observations
- Prevents creation of tokens like `dog.`, `dog?`, `dog!` — reducing token redundancy
- Avoids spurious multi-word tokens like `. Archived from the original on June`
- Leads to cleaner vocabularies and improved token compression

### 📦 Full vs Chunk-Based Encoding
> Full vs chunk-based encoding yields slightly different outputs. Chunk-based approaches avoid merging across regex-based group boundaries. To remain consistent with OpenAI GPT-2’s behavior, we encode chunks separately and join them.

### 🧰 Implementation Design Takeaways
- Respect group boundaries when applying merges
- Avoid reintroducing cross-boundary merges in later BPE iterations
- Keep your tokenizer modular to support experiments like merge visualization, logging, and ablation studies

## 🧗‍♀️ Learning Philosophy
This repository follows the spirit of **learning by building**. It’s not about reusing libraries — it’s about engaging directly with the logic behind tokenization, understanding failure modes, and discovering how merge order, grouping strategies, and encoding techniques affect performance.

> “Start small, start dumb, start moving.” — my motto when stuck, learned through Karpathy’s process.

## 📂 Structure
- `minbpe/tokenizer.py`: core BPE logic (merging, encoding, training), includes basic and regex-based tokenizers
- `train.py`: example code to try tokenizers and look how they train, best place to experiment with different texts, vocabulary sizes, etc
- `play.ipynb`: drafts and analyses
- `tests/`: minimal unit test suit

## 🚧 TODO
- Add visual merge graph tool

## 📜 License
MIT

---

Feel free to fork and play with the internals. The more you tweak, the more you learn. Contributions welcome — especially if you're learning, too!
