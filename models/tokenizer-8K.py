import torch
from transformers import AutoTokenizer
import json


k = 7999

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size if not k else k


file_path = 'tokens.json'
with open(file_path, 'r') as file:
   tokens_counts = json.load(file)


total_tokens = sum(tokens_counts.values()) # Already sorted
if k:
  tokens_used = sum([i for i in tokens_counts.values()][:k])
  top_k_tokens = [i for i in tokens_counts.keys()][:k]# We will only use top k tokens, others will be ignored
  top_k_tokens_dict =  {token: index for index, token in enumerate(top_k_tokens)}
  reversed_top_k_tokens_dict = {value: int(key) for key, value in top_k_tokens_dict.items()}
  
  top_k_tokens.append("50256")
  vocab_size +=1

encode = lambda x: tokenizer(x , return_tensors='pt', padding="max_length", max_length=block_size, truncation=True)['input_ids'].to(device)
decode = lambda x: tokenizer.decode(x, skip_special_tokens=True) 


if k:
  print(f"Used/Total Vocab size: {vocab_size}/{tokenizer.vocab_size}")
  print(f"Used (total accourence of top {k} tokens)/Total Tokens in Data: {tokens_used  / 1e6}/{total_tokens  / 1e6}M | Replaced Tokens (total accourence of {tokenizer.vocab_size - k} tokens): {(total_tokens - tokens_used)  / 1e6}M")
print(f"Vocab size: {vocab_size}")
print(f"Used for our Training: {(n * block_size)  / 1e6}M | Total Tokens (Excluding padded): {total_tokens  / 1e6}M | Total Tokens (Including padded): {(len(data) * block_size) / 1e6}M")
print(f"Examples: {n / 1e6}/{len(data) / 1e6}M")
print(f"Training Batches (of {batch_size} examples): ",len(train_data))
print(f"Validation Batches (of {batch_size} examples): ",len(val_data))
print(f"Epochs: {epochs}, Steps: {max_steps}")
print(f"Calculate Loss every {eval_interval} steps and use {eval_steps} examples")



def encoder(input):
  tokens = encode(input)
  if k:
    tokens = torch.tensor([top_k_tokens_dict.get(str(token.item()), top_k_tokens_dict["50256"]) for token in tokens.view(-1)], device=device).view(tokens.shape)
  return tokens

def decoder(tokens):
  if k:
    for i in range(tokens.shape[0]):
      for j in range(tokens.shape[1]):
        tokens[i][j] = reversed_top_k_tokens_dict[tokens[i][j].item()]
  output = [decode(x) for x in tokens]
  return output