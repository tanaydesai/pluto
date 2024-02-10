import torch
from transformers import AutoTokenizer
import json

class Tokenizer:
  def __init__(self, config, k=None, file_path=None):
    self.k = k
    self.file_path = file_path
    self.tokenizer = AutoTokenizer.from_pretrained(config["name"])
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.vocab_size = self.tokenizer.vocab_size if not self.k else self.k
    self.initialize()

  def get_config(self):
    config = {
        "initl_vocab_size": self.tokenizer.vocab_size,
        "final_vocab_size": self.vocab_size,
        "vocab_size": self.vocab_size,
        "total_tokens": self.total_tokens,
        "total_tokens_used": self.tokens_used if self.k else self.total_tokens,
        "total_unsed_tokens": self.total_tokens - self.tokens_used if self.k else 0
    }
    return config

  def initialize(self):
    with open(self.file_path, 'r') as file:
      tokens_counts = json.load(file)

    self.total_tokens = sum(tokens_counts.values()) # Already sorted

    if self.k:
      self.tokens_used = sum([i for i in tokens_counts.values()][:self.k])
      self.top_k_tokens = [i for i in tokens_counts.keys()][:self.k]# We will only use top k tokens, others will be ignored
      self.top_k_tokens_dict =  {token: index for index, token in enumerate(self.top_k_tokens)}
      self.reversed_top_k_tokens_dict = {value: int(key) for key, value in self.top_k_tokens_dict.items()}  
      self.top_k_tokens.append("50256")
      self.vocab_size +=1

  def encoder(self, input, device="cpu", block_size=256):
    tokens = self.tokenizer(input , return_tensors='pt', padding="max_length", max_length=block_size, truncation=True)['input_ids'].to(device)
    
    if self.k:
      tokens = torch.tensor([self.top_k_tokens_dict.get(str(token.item()), self.top_k_tokens_dict["50256"]) for token in tokens.view(-1)], device=device).view(tokens.shape)

    return tokens

  def decoder(self, tokens):
    if self.k:
      tokens = torch.tensor([[self.reversed_top_k_tokens_dict[token.item()] for token in row] for row in tokens], device=tokens.device)
    
    output = [self.tokenizer.decode(x, skip_special_tokens=True) for x in tokens]

    return output