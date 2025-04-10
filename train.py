"""As in the original minbpe repo,
this file is to check tokenizer/s functionality.
"""
from pathlib import Path
from minbpe import BasicTokenizer, RegexTokenizer

text = open("./tests/taylorswift.txt", 'r', encoding='utf-8').read()
# check training with small vocabulary size
vocab_size = 1024
# create models folder
models_path = Path(__file__).resolve().parent / 'models'
models_path.mkdir(exist_ok=True)
for name, tokenizer in zip(['basic', 'regex'], [BasicTokenizer, RegexTokenizer]):
  t1 = tokenizer()
  t1.train(text, vocab_size, verbose=True)
  # save tokenizer
  t1.save(models_path / name)
  # load and ensure encoding and decoding works
  t2 = tokenizer()
  t2.load(models_path / name)
  print(t2.decode(t1.encode(text)) == text)
