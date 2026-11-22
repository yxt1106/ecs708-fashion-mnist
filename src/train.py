import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


# Reproducibility
SEED_MLP = 45

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED_MLP)
PART2_BATCH_SIZE = 128

def make_part2_loaders(seed):
  generator = torch.Generator()
  generator.manual_seed(seed)

  # Training loader: shuffled
  part2_train_loader = DataLoader(
    train_dataset,
    batch_size=PART2_BATCH_SIZE,
    shuffle=True,
    generator=generator,
    num_workers=0
  )

  # Training evaluation loader: fixed order
  part2_train_eval_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0
  )

  # Validation loader: fixed order
  part2_val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0
  )

  return (
    part2_train_loader,
    part2_train_eval_loader,
    part2_val_loader
  )

# TRaining loop
def train_one_epoch(
  model,
  train_loader,
  criterion,
  optimizer,
  device
):
  model.train()
  for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    # Clear old gradients
    optimizer.zero_grad(set_to_none=True)

    # Forward propagation
    outputs = model(images)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

# Validation Loop
@torch.no_grad()
def evaluate_model(
  model,
  data_loader,
  criterion,
  device
):
  model.eval()
  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  for images, labels in data_loader:
    images = images.to(device)
    labels = labels.to(device)

    # Forward propagation
    outputs = model(images)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Predicted classes
    predictions = outputs.argmax(dim=1)
    batch_size = labels.size(0)
    total_loss += loss.item() * batch_size
    total_correct += (
        predictions == labels
    ).sum().item()
    total_samples += batch_size

  average_loss = total_loss / total_samples
  accuracy = total_correct / total_samples

  return {
    "loss": average_loss,
    "accuracy": accuracy
  }
# Combine dataLoader, training, evaluate into a function
def run_mlp_experiment(
  hidden_sizes=(256, 128),
  learning_rate=0.001,
  seed=SEED_MLP,
  max_epochs=20,
  patience=5,
  min_delta=1e-4
):

  # Reproducibility
  set_seed(seed)

  # Create DataLoaders
  (
    train_loader,
    train_eval_loader,
    val_loader
  ) = make_part2_loaders(seed)

  # Create model
  model = MLP(
    hidden_sizes=hidden_sizes
  ).to(device)

  # Loss function: Chose CrossEntropy to fit multi classification tasks
  criterion = nn.CrossEntropyLoss()

  # Optimizer
  optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
  )

  # Trainable parameter count
  parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
  )

  # Report best data
  best_val_loss = float("inf")
  best_epoch = 0
  best_state = None
  epochs_without_improvement = 0
  history = []

  # Main training loop
  for epoch in range(1, max_epochs + 1):
    # Training
    train_one_epoch(
      model,
      train_loader,
      criterion,
      optimizer,
      device
    )

    # Evaluate training performance
    train_metrics = evaluate_model(
      model,
      train_eval_loader,
      criterion,
      device
    )

    # Evaluate validation performance
    val_metrics = evaluate_model(
      model,
      val_loader,
      criterion,
      device
    )

    # Record metrics
    history.append({
      "epoch": epoch,
      "train_loss": train_metrics["loss"],
      "train_accuracy": train_metrics["accuracy"],
      "val_loss": val_metrics["loss"],
      "val_accuracy": val_metrics["accuracy"]
    })

    # Checkpoint using validation loss
    if val_metrics["loss"] < best_val_loss - min_delta:
      best_val_loss = val_metrics["loss"]
      best_epoch = epoch
      best_state = copy.deepcopy(
          model.state_dict()
      )
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1


    print(
      f"Epoch [{epoch}/{max_epochs}] "
      f"Train Loss: {train_metrics['loss']:.4f} "
      f"Val Loss: {val_metrics['loss']:.4f} "
      f"Train Acc: {train_metrics['accuracy']:.4f} "
      f"Val Acc: {val_metrics['accuracy']:.4f}"
    )


    # Early stopping
    if epochs_without_improvement >= patience:
      print("Early stopping triggered.")
      break


  # Restore best checkpoint
  if best_state is None:
    raise RuntimeError("No valid model checkpoint was saved.")

  model.load_state_dict(best_state)


  history_df = pd.DataFrame(history)

  best_record = history_df.loc[
    history_df["epoch"] == best_epoch
  ].iloc[0]


  return {
    "model": model,
    "history": history_df,
    "parameter_count": parameter_count,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "best_train_accuracy": best_record["train_accuracy"],
    "best_val_accuracy": best_record["val_accuracy"],
    "epochs_trained": len(history_df)
  }
