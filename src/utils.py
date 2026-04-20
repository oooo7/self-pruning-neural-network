import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# =========================
# SPARSITY
# =========================
def compute_sparsity(model, threshold=1e-2):
    total = 0
    zero = 0

    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores).detach()

            total += gates.numel()
            zero += (gates < threshold).sum().item()

    if total == 0:
        return 0

    return zero / total


# =========================
# ACCURACY
# =========================
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# =========================
# GATE DISTRIBUTION
# =========================
def plot_gate_distribution(model, save_path):
    all_gates = []

    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()


# =========================
# LOSS CURVE
# =========================
def plot_training_curve(losses, save_path):
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    plt.close()


# =========================
# TRADEOFF PLOT
# =========================
def plot_tradeoff(lambdas, accuracies, sparsities, save_path):
    plt.figure()
    plt.plot(sparsities, accuracies, marker='o')

    for i in range(len(lambdas)):
        plt.text(sparsities[i], accuracies[i], str(lambdas[i]))

    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Sparsity Trade-off")

    plt.savefig(save_path)
    plt.close()


# =========================
# HARD PRUNING 
# =========================
def apply_hard_pruning(model, threshold=0.5):
    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)

            # Convert to binary mask
            hard_mask = (gates > threshold).float()

            # Apply pruning permanently
            module.weight.data *= hard_mask

            # Freeze gates (optional but recommended)
            module.gate_scores.data = torch.log(hard_mask + 1e-8)