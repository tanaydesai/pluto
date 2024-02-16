import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
import re
from torch.utils.data import DataLoader
from model import GPT2


def load_model(config, path, device='cpu'):
    model = GPT2(config, device=device)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    return model

def load_data(config, batch_size, n, device='cpu'):
    dataset = load_dataset(config.name)
    train_data = DataLoader(dataset["train"][:n]["text"], batch_size=batch_size, shuffle=True, pin_memory=True, pin_memory_device=device)
    val_data = DataLoader(dataset["validation"][:n]["text"], batch_size=batch_size, shuffle=True, pin_memory=True, pin_memory_device=device)

    return train_data, val_data


def clean_string(input_string):
    cleaned_string = re.sub(r'[^\w\s.,]', '', input_string)
    cleaned_string = cleaned_string.replace('\n', '')
    return cleaned_string

@torch.no_grad()
def estimate_loss(model, train_data, val_data, encoder, eval_steps=50):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            data = train_data if split == 'train' else val_data
            tokens = encoder(next(iter(data))[0], max_length=model.block_size, padding="max_length", truncation=True)
            _, loss = model(tokens, tokens)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def plot_losses(losses):
    train_losses = [o['train'] for o in losses if o.get('train') is not None]
    valid_losses = [o['valid'] for o in losses if o.get('valid') is not None]
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.legend()
    plt.show()