"""As in the original minbpe repo,
this file is to check tokenizer/s functionality.
"""
from pathlib import Path
from minbpe import BasicTokenizer

text = open("./tests/taylorswift.txt", 'r', encoding='utf-8').read()
# check training with small vocabulary size
vocab_size = 512
tokenizer = BasicTokenizer()
tokenizer.train(text, vocab_size, verbose=True)
# create models folder
models_path = Path(__file__).resolve().parent / 'models'
models_path.mkdir(exist_ok=True)
# save tokenizer
tokenizer.save(models_path / 'basic')
