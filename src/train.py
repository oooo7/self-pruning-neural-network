import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import PrunableNetwork
from utils import (
    compute_sparsity,
    compute_accuracy,
    plot_gate_distribution,
    plot_training_curve,
    plot_tradeoff,
    apply_hard_pruning
)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# HYPERPARAMETERS
# =========================
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

LAMBDA_LIST = [1e-5, 1e-4, 1e-3]

# =========================
# DATA
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# =========================
# SPARSITY LOSS
# =========================
def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)
            loss += (gates * (1 - gates)).sum()
    return loss


# =========================
# RESULTS DIR
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

all_acc = []
all_sparsity = []

best_acc = 0
best_lambda = None

# =========================
# TRAINING LOOP
# =========================
for LAMBDA in LAMBDA_LIST:

    print(f"\n========== Running for LAMBDA = {LAMBDA} ==========")

    model = PrunableNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            ce_loss = criterion(outputs, labels)

            sp_loss = sparsity_loss(model)

            loss = ce_loss + LAMBDA * sp_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_history.append(running_loss)
        print(f"[INFO] Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f}")

    # Apply hard pruning AFTER training
    apply_hard_pruning(model, threshold=0.5)
    # =========================
    # EVALUATION
    # =========================
    accuracy = compute_accuracy(model, test_loader, device)
    sparsity = compute_sparsity(model)

    print(f"[RESULT] Lambda: {LAMBDA}, Accuracy: {accuracy:.4f}, Sparsity: {sparsity:.4f}")

    all_acc.append(accuracy)
    all_sparsity.append(sparsity)

    # =========================
    # BEST MODEL
    # =========================
    if accuracy > best_acc:
        best_acc = accuracy
        best_lambda = LAMBDA

        torch.save(model.state_dict(),
                   os.path.join(RESULTS_DIR, "best_model.pth"))

    # =========================
    # HARD PRUNING 
    # =========================
    apply_hard_pruning(model, threshold=0.5)

    pruned_sparsity = compute_sparsity(model)
    print(f"[PRUNED] Lambda: {LAMBDA}, Sparsity after pruning: {pruned_sparsity:.4f}")

    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, f"pruned_model_{LAMBDA}.pth"))

    # =========================
    # SAVE RESULTS
    # =========================
    plot_gate_distribution(model, os.path.join(RESULTS_DIR, f"gate_distribution_{LAMBDA}.png"))
    plot_training_curve(loss_history, os.path.join(RESULTS_DIR, f"loss_curve_{LAMBDA}.png"))

# =========================
# FINAL OUTPUTS
# =========================
plot_tradeoff(LAMBDA_LIST, all_acc, all_sparsity,
              os.path.join(RESULTS_DIR, "tradeoff.png"))

print("\n===== FINAL RESULTS =====")
for i in range(len(LAMBDA_LIST)):
    print(f"Lambda: {LAMBDA_LIST[i]} | Accuracy: {all_acc[i]:.4f} | Sparsity: {all_sparsity[i]:.4f}")

print(f"\nBest Model → Lambda: {best_lambda}, Accuracy: {best_acc:.4f}")
print("\nAll experiments completed! Check the 'results' folder.")