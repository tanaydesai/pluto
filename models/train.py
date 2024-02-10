


start_time = time.time()
for epoch in range(epochs):
  print(f"Starting Epoch: {epoch + 1} {'-' * 100}")
  for batch in train_data:
    if steps % eval_interval == 0 or steps == max_steps-1:
      losses = estimate_loss()
      tracked_losses.append(losses)
      print(f"Epoch: {epoch + 1}/{epochs} | Step: {steps}/{max_steps} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")

    tokens = encoder(batch)
    logits, loss = model(tokens, tokens)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    ellapsed_time = time.time() - start_time
    if steps % eval_interval == 0 or steps == max_steps-1:
      print(f"Elapsed time: {ellapsed_time:.2f}")
    steps += 1


print("-" * 80)
plot_losses(tracked_losses)
print("-" * 80)
print("This Model has: ",sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
print("-" * 80)
for i in decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_tokens=300)):
  print(i)