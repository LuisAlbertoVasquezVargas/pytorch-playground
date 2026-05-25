# 05_circle_binary_classification.py

import time

import torch
from torch import nn


def main():
    torch.manual_seed(0)

    # Generate random points in the square [-2, 2] x [-2, 2].
    n = 10_000

    x = torch.rand(n, 2) * 4.0 - 2.0

    # Label rule:
    # inside circle  -> 1
    # outside circle -> 0
    #
    # Circle centered at (0, 0), radius = 1.
    y = ((x[:, 0] ** 2 + x[:, 1] ** 2) <= 1.0).float().reshape(-1, 1)

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10_000

    start_time = time.perf_counter()

    for epoch in range(epochs):
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        correct = (predictions == y).sum().item()
        accuracy = correct / n

    test_points = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [1.5, 0.0],
            [2.0, 2.0],
            [-0.3, 0.4],
        ]
    )

    with torch.no_grad():
        test_logits = model(test_points)
        test_probabilities = torch.sigmoid(test_logits)
        test_predictions = (test_probabilities >= 0.5).float()

    print(f"epochs = {epochs}")
    print(f"running time = {elapsed_time:.6f} seconds")
    print(f"final loss = {loss.item():.12f}")
    print(f"accuracy = {accuracy:.6f}")
    print()

    print("x        | y        | real | probability | predicted")
    print("-" * 57)

    for point, prob, pred in zip(test_points, test_probabilities, test_predictions):
        px = point[0].item()
        py = point[1].item()

        real = 1.0 if px * px + py * py <= 1.0 else 0.0

        print(
            f"{px:8.4f} | "
            f"{py:8.4f} | "
            f"{real:4.0f} | "
            f"{prob.item():11.6f} | "
            f"{pred.item():9.0f}"
        )


if __name__ == "__main__":
    main()
