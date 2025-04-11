"""
Implements basic and regex BPE tokenizer as in
https://github.com/karpathy/minbpe/blob/master/exercise.md
"""

import json
import regex as re
from pathlib import Path


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Helper functions
def get_stats(ids, stats=None):
  """Given ids (tokens) computes
  token consecutive pairs frequencies
  """
  if stats is None:
    stats = {}
  for pair in zip(ids, ids[1:]):
    stats[pair] = stats.get(pair, 0) + 1
  return stats

def merge(ids, pair, idx):
  """Iterates through ids and merges specified
  pair `pair` into new index `idx`
  """
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# ----------------------------------------------------
 
# Most basic BPE Tokenizer 
class BasicTokenizer():
  def __init__(self):
    self._build_clean_state()

  def _build_clean_state(self):
    self.merges = {}
    self.vocab = {i:bytes([i]) for i in range(256)}

  def train(self, text, vocab_size, verbose=False):
    """Train bpe tokenizer on string `text`, trained tokenizer
    will have vocabulary size of `vocab_size`. If verbose,
    print merges and final stats
    """
    # ensure clean state [e.g. what if we train after we've trained]
    num_iters = vocab_size - 256
    tokens = list(text.encode('utf-8'))
    old_len = len(tokens)
    for i in range(num_iters):
      stats = get_stats(tokens)
      pair = max(stats, key=stats.get) # type: ignore
      idx = 256 + i
      tokens = merge(tokens, pair, idx)
      self.merges[pair] = idx
      self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
      if verbose:
        # visualize during training, print bytes
        print(self.vocab[pair[0]], '+', self.vocab[pair[1]], '->', self.vocab[idx])
    new_len = len(tokens)
    if verbose:
      print(f'Length of tokens before: {old_len}')
      print(f'Length of tokens after: {new_len}')
      print(f'Compression ratio: {old_len/new_len:.3}X')

  def encode(self, text):
    """Given string encode it to tokens"""
    tokens = list(text.encode('utf-8'))
    while True:
      # want to find tokens pair which is in merges and has lowest rank
      stats = get_stats(tokens)
      try: # we found at least one mergeable pair and took one with the lowest rank
        idx, pair = min(((self.merges[p], p) for p in stats if p in self.merges))
        tokens = merge(tokens, pair, idx)
      except ValueError:
      # we are breaking, as we didn't find any pair in tokens which is also in merges
        break
    return tokens

  def decode(self, ids):
    """Given encoded ids, return string"""
    text = b''.join(self.vocab[i] for i in ids)
    return text.decode('utf-8', errors='replace')

  def save(self, filepath):
    """Saves the state of tokenizer into file"""
    model_path = Path(filepath).with_suffix('.model')
    # save only pairs, they are ordered in modern python
    model_path.write_text(json.dumps(list(self.merges)))

  def load(self, filepath):
    """Loads the state of tokenizer from filepath.
    Intended to use after training and saving,
    before attempting to encode/decode
    """
    model_path = Path(filepath).with_suffix('.model')
    pairs = json.loads(model_path.read_text())
    # let's ensure clean state
    self._build_clean_state()
    # recreate merges and vocab
    for idx, pair in enumerate(pairs, start=256):
      self.merges[tuple(pair)] = idx
      self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

# Regex Tokenizer using GPT4 split pattern
# does not allow token merges across character categories' borders
# only train is different from basic tokenizer
class RegexTokenizer(BasicTokenizer):
  def __init__(self):
    super().__init__()
    self.pattern = re.compile(GPT4_SPLIT_PATTERN) 
  
  def train(self, text, vocab_size, verbose=False):
    groups = self.pattern.findall(text)
    tokens = [list(g.encode('utf-8')) for g in groups]
    num_iters = vocab_size - 256
    old_len = sum(len(t) for t in tokens)

    for i in range(num_iters):
      stats = {}
      for tg in tokens:
        stats = get_stats(tg, stats) # accumulate stats
      pair = max(stats, key=stats.get) # type: ignore
      idx = 256 + i
      tokens = [merge(t, pair, idx) for t in tokens]
      self.merges[pair] = idx
      self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
      if verbose:
        # visualize during training, print bytes
        print(self.vocab[pair[0]], '+', self.vocab[pair[1]], '->', self.vocab[idx])

    new_len = sum(len(t) for t in tokens)
    if verbose:
      print(f'Length of tokens before: {old_len}')
      print(f'Length of tokens after: {new_len}')
      print(f'Compression ratio: {old_len/new_len:.3}X')

  def encode(self, text):
    # encode chunks -> return joined
    chunks = self.pattern.findall(text)
    tokens = []
    for chunk in chunks:
      tokens.extend(super().encode(chunk))
    return tokens
