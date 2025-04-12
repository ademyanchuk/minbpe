"""As in the original minbpe repo,
this file is to check tokenizer/s functionality.
"""
from pathlib import Path
from minbpe import BasicTokenizer, RegexTokenizer

text = open("./tests/taylorswift.txt", 'r', encoding='utf-8').read()
# check training with small vocabulary size
vocab_size = 1024
base_path = Path(__file__).resolve().parent
# create models folder
models_path = base_path / 'models'
models_path.mkdir(exist_ok=True)
# create verbose output files folder
out_path = base_path / 'out' 
out_path.mkdir(exist_ok=True)
# train, save, load, check identity for both tokenizers
for name, tokenizer in zip(['basic', 'regex'], [BasicTokenizer, RegexTokenizer]):
  t1 = tokenizer()
  with (out_path / f'{name}.txt').open(mode='w') as file:
    t1.train(text, vocab_size, verbose=True, file=file) 
  # save tokenizer
  t1.save(models_path / name)
  # load and ensure encoding and decoding works
  t2 = tokenizer()
  t2.load(models_path / name)
  print(t2.decode(t1.encode(text)) == text)
