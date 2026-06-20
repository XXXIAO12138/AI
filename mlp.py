import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1) Device: prefer MPS (Mac) -> CUDA -> CPU
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# 2) Make some sample data (binary classification)
torch.manual_seed(0)
N = 2000
in_dim = 20

X = torch.randn(N, in_dim)
true_w = torch.randn(in_dim, 1)
logits = X @ true_w + 0.1 * torch.randn(N, 1)
y = (logits > 0).float()  # labels: 0/1

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3) Minimal MLP
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # output logits
        )

    def forward(self, x):
        return self.net(x)

model = MLP(in_dim).to(device)

# 4) Loss + optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 5) Train loop
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred_logits = model(xb)
        loss = criterion(pred_logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

        # accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(pred_logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()

    avg_loss = total_loss / N
    acc = correct / total
    print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | acc={acc:.4f}")

# 6) Quick inference on a few samples
model.eval()
with torch.no_grad():
    xb, yb = next(iter(loader))
    xb = xb.to(device)
    logits = model(xb)
    probs = torch.sigmoid(logits).squeeze(-1).cpu()
    print("Example probs (first 5):", probs[:5].tolist())