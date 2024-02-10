from model import GPT2
import torch
from train import Trainer
from tokenizer import Tokenizer
from utils import *
from config import config

model_name= "gpt-1M"
train_data, val_data = load_data(config.data,config[model_name]["batch_size"], config[model_name]["n"], device="cpu")
tokenizer = Tokenizer(config.tokenizer, k=config[model_name]["k"], file_path="tokens.json")
print(tokenizer.get_config())

model_config = config[model_name]
model = GPT2(model_config)
optim = torch.optim.Adam(model.parameters(), lr=model_config.lr)

trainer = Trainer(model, optim, train_data, val_data, tokenizer.encoder)
tracked_losses = trainer.train(epochs=1, max_steps=1, eval_interval=200, eval_steps=50)
plot_losses(tracked_losses)