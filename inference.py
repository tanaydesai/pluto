import sys
sys.path.append('./models') 
from model import GPT2
import torch
from tokenizer import Tokenizer
from config import config
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name= "gpt-1M"
path = "model-1M.bin"
model_config = config[model_name]

tokenizer = Tokenizer(config.tokenizer, k=model_config.k, file_path="tokens.json", device=device)

model = load_model(model_config, path, device=device)

unconditional = torch.zeros((1, 1), dtype=torch.long, device=device)
prompt = "Elon told his mom"

output1 = model.generate(unconditional, max_tokens=200, temperature=1, top_k=None)
output2 = model.generate(tokenizer.encoder(prompt), max_tokens=200, temperature=1, top_k=None)

print(clean_string(tokenizer.decoder(output1)[0]))
print(clean_string(tokenizer.decoder(output2)[0]))