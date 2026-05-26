# 06_multiclass_classification.py

import time

import torch
from torch import nn


torch.manual_seed(0)


def make_data(n):
    x = torch.rand(n, 2) * 4.0 - 2.0

    y = torch.zeros(n, dtype=torch.long)

    y[x[:, 0] < 0.0] = 0
    y[(x[:, 0] >= 0.0) & (x[:, 1] >= 0.0)] = 1
    y[(x[:, 0] >= 0.0) & (x[:, 1] < 0.0)] = 2

    return x, y


n = 2000
epochs = 10000
lr = 0.01

x, y = make_data(n)

model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

start = time.perf_counter()

for epoch in range(epochs):
    logits = model(x)

    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

end = time.perf_counter()

with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
    predicted = probabilities.argmax(dim=1)

    accuracy = (predicted == y).float().mean()

print(f"epochs = {epochs}")
print(f"running time = {end - start:.6f} seconds")
print(f"final loss = {loss.item():.12f}")
print(f"accuracy = {accuracy.item() * 100:.2f}%")

print()
print("x        y        | real class | predicted | probabilities")
print("----------------------------------------------------------")

for i in range(10):
    print(
        f"{x[i, 0].item(): 8.4f} "
        f"{x[i, 1].item(): 8.4f} |"
        f"{y[i].item(): 11d} |"
        f"{predicted[i].item(): 10d} | "
        f"{probabilities[i].tolist()}"
    )
