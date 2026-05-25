<!-- README.md -->

# pytorch-playground

Small PyTorch learning repository.

The goal of this repository is to study PyTorch from zero using small runnable examples.

---

## Environment

This project uses `micromamba`.

Create and activate the environment:

```bash
micromamba create -n venv python=3.11 -y
micromamba activate venv
````

Install PyTorch CPU version:

```bash
micromamba install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.rand(2, 3))"
```

Current verified output:

```text
2.5.1
tensor([[0.9755, 0.8132, 0.5048],
        [0.0936, 0.4823, 0.1318]])
```

---

## First Goal

Start with the PyTorch basics:

* tensors
* tensor operations
* gradients
* simple models
* training loops

---

## LWC

```bash
find . -maxdepth 2 -not -path '*/.*' -type f \( -name "*.py" -o -name "*.md" -o -name "*.cpp" -o -name "*.hpp" \) -exec echo "--- {} ---" \; -exec cat {} \;
```

