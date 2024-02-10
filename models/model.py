import torch
import torch.nn as nn
from torch.nn import functional as F


n = 1200000
k = 7999
vocab_size = k + 1
batch_size = 64
block_size = 256
max_pos_n_embed = 2048
lr = 2e-3
n_layer = 8
n_head = 16
n_embed = 320
dropout = 0.2
epochs = 1
beta1 = 0.9
beta2 = 0.95

max_steps = epochs * round(n / batch_size)
eval_interval = 200
eval_steps = 50
steps = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # (T, T)
    self.dropout = nn.Dropout(dropout)

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
  

class MultiHead(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj  = nn.Linear(head_size * num_heads, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    out = torch.concat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out
  

class FeedForward(nn.Module):
  def __init__(self, n_embed):
   super().__init__()
   self.layers = nn.Sequential(
        nn.Linear(n_embed, 4 * n_embed),
        nn.GELU(),
        nn.Linear(4 * n_embed, n_embed),
        nn.Dropout(dropout),
    )

  def forward(self,x):
    return self.layers(x)
  


class Block(nn.Module):
  def __init__(self,n_embed, n_head):
    super().__init__()
    head_size = n_embed // n_head
    self.sa_heads = MultiHead(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
  

class GPT2(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedings = nn.Embedding(vocab_size, n_embed)
    self.position_embedings = nn.Embedding(max_pos_n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)
    self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
    self.ln_final = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def get_parameters(self):
    return sum(p.numel() for p in self.parameters())

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embed = self.embedings(idx) # (B, T, C)
    position_embed = self.position_embedings(torch.arange(T,  device=device)) # (T, C)
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
      idx_cond = idx[:, -block_size:]
      logits, _ = self(idx_cond) # (B, T, C)
      logits = logits[:, -1, :]  / temperature # (B, C)
      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
      probs = F.softmax(logits, dim=-1) # Softmax Independently for C dim
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.concat((idx, idx_next), dim=1) # (B, T+1)
    return idx