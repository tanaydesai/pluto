from utils import *

class Trainer:
  def __init__(self, model, optimizer, train_data, val_data, encoder):
    self.model = model
    self.optimizer = optimizer
    self.train_data = train_data
    self.val_data = val_data
    self.encoder = encoder

  def train(self, epochs, max_steps, eval_interval=200, eval_steps=50):
    steps = 0
    tracked_losses = []

    for epoch in range(epochs):
      print(f"Starting Epoch: {epoch + 1} {'-' * 100}")
      for batch in self.train_data:
        if steps % eval_interval == 0 or steps == max_steps-1:
          losses = estimate_loss(self.model, self.train_data, self.val_data, self.encoder, eval_steps)
          tracked_losses.append(losses)
          print(f"Epoch: {epoch + 1}/{epochs} | Step: {steps}/{max_steps} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")

        tokens = self.encoder(batch)
        _, loss = self.model(tokens, tokens)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        steps += 1

    return tracked_losses