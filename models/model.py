import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
  def __init__(self, config, head_size):
    super().__init__()
    self.key = nn.Linear(config.n_embed, head_size, bias=False)
    self.query = nn.Linear(config.n_embed, head_size, bias=False)
    self.value = nn.Linear(config.n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))  # (T, T)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B, T, C)
    q = self.query(x) # (B, T, C)
    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, C) X (B, C, T) --> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)  # (B,T,C)
    out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
    return out
  

class MultiHeadAttention(nn.Module):
  def __init__(self, config, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
    self.proj  = nn.Linear(head_size * config.n_head, config.n_embed)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self,x):
    out = torch.concat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out
  

class FeedForward(nn.Module):
  def __init__(self, config):
   super().__init__()
   self.layers = nn.Sequential(
        nn.Linear(config.n_embed, 4 * config.n_embed),
        nn.GELU(),
        nn.Linear(4 * config.n_embed, config.n_embed),
        nn.Dropout(config.dropout),
    )

  def forward(self,x):
    return self.layers(x)
  

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    head_size = config.n_embed // config.n_head
    self.sa_heads = MultiHeadAttention(config, head_size)
    self.ffwd = FeedForward(config)
    self.ln1 = nn.LayerNorm(config.n_embed)
    self.ln2 = nn.LayerNorm(config.n_embed)

  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
  

class GPT2(nn.Module):
  def __init__(self, config, device='cpu'):
    super().__init__()
    self.device = device
    self.block_size = config.block_size
    self.embedings = nn.Embedding(config.vocab_size, config.n_embed)
    self.position_embedings = nn.Embedding(config.max_pos_n_embed, config.n_embed)
    self.dropout = nn.Dropout(config.dropout)
    self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
    self.ln_final = nn.LayerNorm(config.n_embed)
    self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

  def get_parameters(self):
    return sum(p.numel() for p in self.parameters())

  def save(self, path):
    torch.save(self.state_dict(), path)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embed = self.embedings(idx) # (B, T, C)
    position_embed = self.position_embedings(torch.arange(T,  device=self.device)) # (T, C)
    x = token_embed + position_embed # (B, T, C)
    x = self.dropout(x) # (B, T, C)
    x = self.blocks(x) # (B, T, C)
    x = self.ln_final(x) # (B, T, C)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      logits = logits[..., :-1, :].contiguous()
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets[..., 1:].contiguous().view(-1), ignore_index=50256)
    return logits, loss

  def generate(self, idx, max_tokens, temperature=1.0, top_k=None):
    # idx is (B, T)
    for _ in range(max_tokens):
      idx_cond = idx[:, -self.block_size:]
      logits, _ = self(idx_cond) # (B, T, C)
      logits = logits[:, -1, :]  / temperature # (B, C)
      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
      probs = F.softmax(logits, dim=-1) # Softmax Independently for C dim
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.concat((idx, idx_next), dim=1) # (B, T+1)
    return idx