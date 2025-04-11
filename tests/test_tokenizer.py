from pathlib import Path
import pytest

from minbpe import BasicTokenizer, RegexTokenizer

# Borrowed from original minbpe repo

# strings to test
strings = [
  "",
  "!",
  "hello world!!!? (안녕하세요!) lol123 😉",
  "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
]


test_text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."

def unpack(text):
  # FILE: special case to not print the entire content of the file,
  # while running tests `pytest -v .`
  if text.startswith('FILE:'):
     file_path = Path(__file__).resolve().parent
     text = (file_path / text[5:]).read_text()
  return text


#----------------------------------------------------------------

# test encode/decode
@pytest.mark.parametrize("text", strings)
@pytest.mark.parametrize("tokenizer", [BasicTokenizer, RegexTokenizer])
def test_encode_decode(text, tokenizer):
   text = unpack(text)
   t = tokenizer()
   ids = t.encode(text) 
   assert t.decode(ids) == text

# minor test of train as per wiki
@pytest.mark.parametrize("tokenizer", [BasicTokenizer, RegexTokenizer])
def test_wikipedia_example(tokenizer):
    """
    Quick unit test, following along the Wikipedia example:
    https://en.wikipedia.org/wiki/Byte_pair_encoding

    According to Wikipedia, running bpe on the input string:
    "aaabdaaabac"

    for 3 merges will result in string:
    "XdXac"

    where:
    X=ZY
    Y=ab
    Z=aa

    Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
    so Z will be 256, Y will be 257, X will be 258.

    So we expect the output list of ids to be [258, 100, 258, 97, 99]
    """
    t = tokenizer()
    text = "aaabdaaabac"
    t.train(text, 256 + 3)
    ids = t.encode(text)
    assert ids == [258, 100, 258, 97, 99]
    assert t.decode(t.encode(text)) == text

# test save/load
@pytest.mark.parametrize("tokenizer", [BasicTokenizer, RegexTokenizer])
def test_save_load(tokenizer, tmp_path):
  t = tokenizer()
  t.train(test_text, 256 + 64)
  ids = t.encode(test_text)
  t.save(tmp_path / "tok")
  # init and load
  t = tokenizer()
  t.load(tmp_path / "tok")
  ids_new = t.encode(test_text)
  # check we can encode and decode the same thing
  assert ids == ids_new
  assert t.decode(ids_new) == test_text