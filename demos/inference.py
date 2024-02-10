from models.model import GPT2
import torch
from models.tokenizer import Tokenizer
from models.config import config
from models.utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name= "gpt-1M"
path = "model-1M.pth"
model_config = config[model_name]

tokenizer = Tokenizer(config.tokenizer, k=model_config.k, file_path="tokens.json", device=device)

model = load_model(model_config, path, device=device)

prompt1 = torch.zeros((1, 1), dtype=torch.long, device=device)
prompt2 = "Elon told his mom"

output1 = model.generate(prompt1, max_tokens=200, temperature=1, top_k=None)
output2 = model.generate(tokenizer.encoder(prompt2), max_tokens=200, temperature=1, top_k=None)

print(tokenizer.decoder(output1)[0])
print(tokenizer.decoder(output2)[0])