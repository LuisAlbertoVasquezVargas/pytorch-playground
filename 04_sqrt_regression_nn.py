# 04_sqrt_regression_nn.py

import time

import torch
from torch import nn


def main():
    torch.manual_seed(0)

    # Original problem:
    # y = sqrt(x), where x is in [0, 100]
    x_real = torch.linspace(0.0, 100.0, 1000).reshape(-1, 1)
    y_real = torch.sqrt(x_real)

    # Normalize both input and output:
    # x_norm is in [0, 1]
    # y_norm is in [0, 1]
    x = x_real / 100.0
    y = y_real / 10.0

    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100_000

    start_time = time.perf_counter()

    for epoch in range(epochs):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    test_x_real = torch.tensor([[4.0], [9.0], [16.0], [25.0], [64.0], [100.0]])
    test_x = test_x_real / 100.0

    with torch.no_grad():
        test_pred_norm = model(test_x)
        test_pred_real = test_pred_norm * 10.0

    print(f"epochs = {epochs}")
    print(f"running time = {elapsed_time:.6f} seconds")
    print(f"final normalized loss = {loss.item():.12f}")
    print()

    print("x        | real sqrt(x) | predicted")
    print("-" * 38)

    for xi, pi in zip(test_x_real, test_pred_real):
        real = torch.sqrt(xi).item()
        pred = pi.item()

        print(f"{xi.item():8.4f} | {real:12.6f} | {pred:9.6f}")


if __name__ == "__main__":
    main()
