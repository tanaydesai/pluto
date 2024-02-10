import torch
import matplotlib.pyplot as plt



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            data = train_data if split == 'train' else val_data
            tokens = encoder(next(iter(data))[0])
            logits, loss = model(tokens, tokens)
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