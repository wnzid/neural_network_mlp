import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
df = pd.read_csv("dataset_039.csv")

X = df.drop("target", axis=1).values
y = df["target"].values

# If labels are not 0,1,2,... convert them
unique_labels = np.unique(y)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y = np.array([label_to_index[label] for label in y], dtype=np.int64)

num_classes = len(unique_labels)
input_size = X.shape[1]

output_dir = "mlp_results"
os.makedirs(output_dir, exist_ok=True)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 4. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# 5. Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 6. Training function
def train_model(hidden1, hidden2, learning_rate, batch_size, epochs, model_name):
    model = MLP(input_size, hidden1, hidden2, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    n = X_train.shape[0]

    for epoch in range(epochs):
        permutation = torch.randperm(n)

        epoch_loss = 0.0
        model.train()

        for i in range(0, n, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train[indices]
            batch_y = y_train[indices]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (n / batch_size)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"{model_name} | Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    # Save loss curve image
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title(f"Loss Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, f"{model_name}_loss.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)

    y_true = y_test.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"Results for {model_name}")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Save trained weights
    weights_path = os.path.join(output_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), weights_path)

    print(f"Weights saved as: {weights_path}")
    print(f"Loss curve saved as: {loss_plot_path}\n")

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "losses": losses
    }

# 7. Repeat training 3 times
results = []

results.append(train_model(
    hidden1=64,
    hidden2=32,
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    model_name="mlp_run_1"
))

results.append(train_model(
    hidden1=128,
    hidden2=64,
    learning_rate=0.0005,
    batch_size=32,
    epochs=120,
    model_name="mlp_run_2"
))

results.append(train_model(
    hidden1=64,
    hidden2=16,
    learning_rate=0.005,
    batch_size=64,
    epochs=80,
    model_name="mlp_run_3"
))

# 8. Compare final results
print("\nFINAL COMPARISON")
print("=" * 50)
for r in results:
    print(
        f"{r['model_name']}: "
        f"Accuracy={r['accuracy']:.4f}, "
        f"Precision={r['precision']:.4f}, "
        f"Recall={r['recall']:.4f}, "
        f"F1={r['f1']:.4f}"
    )