# 03_linear_regression_scratch.py

import time

import torch


def main():
    # y = 2x + 1

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    lr = 0.01
    epochs = 600_000

    start_time = time.perf_counter()

    for epoch in range(epochs):
        pred = x * w + b
        loss = ((pred - y) ** 2).mean()

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"epochs = {epochs}")
    print(f"running time = {elapsed_time:.6f} seconds")
    print(f"final loss = {loss.item():.12f}")
    print(f"final w = {w.item():.12f}")
    print(f"final b = {b.item():.12f}")


if __name__ == "__main__":
    main()
