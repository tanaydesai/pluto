# Introduction

[Name] is a series of small language models of 1M, 15M and 30M parameters based on the decoder-only transformer architecture similar to gpt-2. This project aims to promote building of coherent english-generating language models by students and individuals at a very low cost, try to reproduce the contents of the TinyStories paper and demonstrate picking only the top 8k most occuring tokens from our vocabulary, based on appearance instead of the usual 50256 `vocab_size` that come with the `EleutherAI/gpt-neo-125M` tokenizer, as per the paper, in my presumed way.

The main files are structured something like this:

- `model.py` includes the transformer model architecture.
- `tokenizer.py` includes a simple class to load our `Tokenizer` object and the tokenizer from HF with an option to load only the top-k tokens in our vocabulary.
- `train.py` includes the class to train the model
- `demos` includes notebooks for inference and training the model.

Support functions like `plot_lossess` and `estimate_loss` for the training phase are in `utils.py`. If you want to see more generations from the models, see  `generate.txt`.


# Preprocessing

According to the TinyStories paper, the authors mentioned taking the top 8k most occuring tokens out of the 50k token vocabulary. So our `Tokenizer` object also includes an optional `k` parameter that we can set to any number to include only the top-k tokens in our tokenizer. The `roneneldan/TinyStories` dataset contains around 476.6M total tokens and 475.5M tokens if we take top-8k most occuring tokens from the vocabulary. Which is pretty neat! This means around 42257/50256 tokens do not even occur that much in our dataset to make a huge difference, so we might as well remove them. As you will see this makes no significant effect on the quality of the model, only reducing the size significantly!

- Getting Top K tokens


# Model Cards
- Architecture
- parameters
- hyperparams


# Training
- loss, etc
- images
- hyperparams
- gpu, time, evals
- cost


# Result
- model 1 = unconditional & prompt
- model 2 = unconditional & prompt
- model 3 = unconditional & prompt
- paper models


# Conclusion
- opinion on small models
- high q data
- etc


# Refrences
- Several Papers
- Datasets
- GitHub Repos
- Andrej Karpaty stuff