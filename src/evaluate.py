import torch
import os
import matplotlib.pyplot as plt


def plot_losses(losses, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(save_path)
    plt.close()

    print("Loss plot saved at:", save_path)