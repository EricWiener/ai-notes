---
tags:
  - flashcards
source: https://huggingface.co/docs/transformers/main_classes/tokenizer
summary: The tokenizer can be used to encode text into token ints. It can also handle things like padding and truncating sequences to form a batch.
publish: true
---
The typical base class you are using when using a `Tokenizer` is `PreTrainedTokenizerBase`. The main method for tokenizers is [`__call__`](https://github.com/huggingface/transformers/blob/6ce6d62b6f20040129ec9831e7c4f6576402ea42/src/transformers/tokenization_utils_base.py#L2509) which is the "method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences."

It makes a call to [`_call_one`](https://github.com/huggingface/transformers/blob/6ce6d62b6f20040129ec9831e7c4f6576402ea42/src/transformers/tokenization_utils_base.py#L2597) which calls `batch_encode_plus` or `encode_plus` which then call `_batch_encode_plus` or `_encode_plus` respectively. This methods are not implemented in `PreTrainedTokenizerBase` and are implemented in the sub-classes.

For instance, `PreTrainedTokenizerFast` implements [`_batch_encode_plus`](https://github.com/huggingface/transformers/blob/4e244b8817356d8a090b80185e8e98a865871e91/src/transformers/tokenization_utils_fast.py#L390) and makes use of `_batch_encode_plus` in it's implementation of `_encode_plus`. 

Both `_batch_encode_plus` and `_encode_plus` accept the following as arguments:
- `max_length`
- `padding_strategy`
- `truncation_strategy`
- etc.

However, you should not specify `padding_strategy` and `truncation_strategy` directly as these are calculated internally via [`_get_padding_truncation_strategies`](https://github.com/huggingface/transformers/blob/6ce6d62b6f20040129ec9831e7c4f6576402ea42/src/transformers/tokenization_utils_base.py#L2370). Instead, you should specify `padding` and `truncation` as seen in the arguments to [`__call__`](https://github.com/huggingface/transformers/blob/6ce6d62b6f20040129ec9831e7c4f6576402ea42/src/transformers/tokenization_utils_base.py#L2518).

### Using a tokenizer
You can initialize a tokenizer with:
```python
model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

You can then use a tokenizer with:
```python
sentences = [
    "Tell me a joke about Lion.",
    "Tell me a joke about Lion that is funny.",
    "Tell me a joke about Dogs that is funny.",
    "Tell me a joke about Lion that is funny."
]
encoded = tokenizer(sentences, truncation=True, padding=True, max_length=10, return_tensors="pt").to("mps")
```

The expected output looks like:
```
Tell me a joke about Lion.</s><pad>
Tell me a joke about Lion that is</s>
Tell me a joke about Dogs that</s>
Tell me a joke about Lion that is</s>
```

**Note:** You need to specify `truncation`, `padding`, `max_length`, and `return_tensors` when you do `tokenizer.__call__()`. You are also able to pass them as arguments to `__init__` since HuggingFace allows passing arbitrary values which are then stored as `self.init_kwargs` but these are not used when executing `__call__()`.

### Working with pairs of sequences
The tokenizer allows you to tokenize a sequence, a batch of sequences, a pair of sequences, or a batch of pairs of sequences.

> [!QUESTION] Why is it common to work with pairs of sequences?
> You often work with pairs of sequences in NLP for text2text tasks like translation, question answering, summarization, etc.

**You can process a pair of sequences as follows:**
```python
sentences = [
    "Tell me a joke about Lion.",
    "Tell me a joke about Lion that is funny.",
]
encoded = tokenizer(*sentences, truncation="only_second", padding=True, max_length=15, return_tensors="pt").to("mps")
```

Which when decoded results in:
```
Tell me a joke about Lion.</s> Tell me a joke</s>
```

**You can process a batch of pairs of sequences as follows:**
```python
first_sentences = ["Hello, I'm a science student.", "I love studying planets."]
second_sentences = ["I'm applying for a Ph.D. in Astronomy.", "Saturn is my favorite."]

# Tokenizing batch of sentence pairs
encoded = tokenizer(first_sentences, second_sentences, padding=True, truncation=True, return_tensors="pt")
```

which results in:
```
Hello, I'm a science student.</s> I'm applying for a Ph.D. in Astronomy.</s>
I love studying planets.</s> Saturn is my favorite.</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
```

**Truncation strategies for pairs of sequences**:
The `truncation` argument controls truncation. It can be a boolean or a string. When dealing with pairs of sequences you can handle the truncation in special ways:
- `True` or `'longest_first'`: truncate to a maximum length specified by the `max_length` argument or the maximum length accepted by the model if no `max_length` is provided (`max_length=None`). This will truncate token by token, removing a token from the longest sequence in the pair until the proper length is reached.
- `'only_second'`: truncate to a maximum length specified by the `max_length` argument or the maximum length accepted by the model if no `max_length` is provided (`max_length=None`). This will only truncate the second sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `'only_first'`: truncate to a maximum length specified by the `max_length` argument or the maximum length accepted by the model if no `max_length` is provided (`max_length=None`). This will only truncate the first sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `False` or `'do_not_truncate'`: no truncation is applied. This is the default behavior.
[Source](https://huggingface.co/docs/transformers/pad_truncation)