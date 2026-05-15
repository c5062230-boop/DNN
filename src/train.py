# src/train.py

import torch
import torch.nn as nn
import os


def train_model(
    model,
    dataloader,
    epochs=2,
    lr=1e-4,
    log_path="results/training_log.txt"
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.MSELoss()

    losses = []

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as log_file:

        total_steps = len(dataloader)

        for epoch in range(epochs):

            model.train()

            epoch_loss = 0

            print(f"\nEpoch {epoch+1}/{epochs}")

            for batch in dataloader:

                images = batch["images"]

                if images.dim() == 4:
                    images = images.unsqueeze(0)

                images = images.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                target = torch.zeros_like(outputs).to(device)

                loss = criterion(outputs, target)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / total_steps

            losses.append(avg_loss)

            log_line = f"Epoch {epoch+1}, Loss: {avg_loss:.6f}"

            print(log_line)

            log_file.write(log_line + "\n")

    return losses