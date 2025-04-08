"""
Implements basic BPE tokenizer as in
https://github.com/karpathy/minbpe/blob/master/exercise.md#step-1
"""

# Helper functions

def get_stats(ids):
  """Given ids (tokens) computes
  token consecutive pairs frequencies
  """
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

class BasicTokenizer():
  def __init__(self):
    self.merges = {}
    self.vocab = {i:bytes([i]) for i in range(256)}

  def train(self, text, vocab_size, verbose=False):
    num_iters = vocab_size - 256
    tokens = list(text.encode('utf-8'))
    old_len = len(tokens)
    for i in range(num_iters):
      stats = get_stats(tokens)
      pair = max(stats, key=stats.get)
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
    pass
  def decode(self, ids):
    pass

if __name__ == "__main__":
  text = open("./tests/taylorswift.txt", 'r', encoding='utf-8').read()
  vocab_size = 512
  tokenizer = BasicTokenizer()
  tokenizer.train(text, vocab_size, verbose=True)
