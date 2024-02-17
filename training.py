import sys
sys.path.append('./models') 
from model import GPT2
import torch
from train import Trainer
from tokenizer import Tokenizer
from utils import *
from config import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name= "gpt-1M"
model_config = config[model_name]

train_data, val_data = load_data(config.data, model_config.batch_size, model_config["n"], device=device)
tokenizer = Tokenizer(config.tokenizer, k=model_config.k, file_path="tokens.json", device=device)

model = GPT2(model_config, device=device)
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=model_config.lr)

trainer = Trainer(model_config, model, optim, train_data, val_data, tokenizer.encoder)
tracked_losses = trainer.train(epochs=1, eval_interval=200, eval_steps=50)
model.save("model-1M.bin")

print(tokenizer.get_config())
print(model.get_parameters())
plot_losses(tracked_losses)