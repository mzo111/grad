"""Train an MLP on MNIST with the grad engine and report test accuracy.

Usage: python examples/mnist.py [--epochs 5] [--batch-size 128] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grad import Tensor, losses, nn, optim  # noqa: E402

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MEAN, STD = 0.1307, 0.3081  # standard MNIST pixel statistics (on the [0, 1] scale)


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        path = data_dir / name
        if path.exists():
            continue
        last_error: Exception | None = None
        for mirror in MIRRORS:
            url = mirror + name
            try:
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, path)
                break
            except OSError as e:  # try the next mirror
                last_error = e
                path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"could not download {name}") from last_error


def read_idx(path: Path) -> np.ndarray:
    """Parse an IDX file: big-endian header (magic, dims...) then uint8 payload."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download(data_dir)
    x_train = read_idx(data_dir / FILES["train_images"])
    y_train = read_idx(data_dir / FILES["train_labels"])
    x_test = read_idx(data_dir / FILES["test_images"])
    y_test = read_idx(data_dir / FILES["test_labels"])

    def prep(x: np.ndarray) -> np.ndarray:
        x = x.reshape(len(x), -1).astype(np.float64) / 255.0
        return (x - MEAN) / STD

    return prep(x_train), y_train.astype(np.int64), prep(x_test), y_test.astype(np.int64)


def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, chunk: int = 2000) -> float:
    correct = 0
    for i in range(0, len(x), chunk):
        logits = model(Tensor(x[i : i + chunk])).data
        correct += int((logits.argmax(axis=1) == y[i : i + chunk]).sum())
    return correct / len(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x_train, y_train, x_test, y_test = load_mnist(args.data_dir)
    print(f"train {x_train.shape}, test {x_test.shape}")

    model = nn.Sequential(
        nn.Linear(784, args.hidden, rng=rng),
        nn.ReLU(),
        nn.Linear(args.hidden, 10, rng=rng),
    )
    opt = optim.Adam(model.parameters(), lr=args.lr)

    n = len(x_train)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        total_loss, batches = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            loss = losses.softmax_cross_entropy(model(Tensor(x_train[idx])), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1
        acc = accuracy(model, x_test, y_test)
        print(
            f"epoch {epoch}/{args.epochs}  train loss {total_loss / batches:.4f}  "
            f"test acc {acc * 100:.2f}%  ({time.perf_counter() - t0:.1f}s)"
        )
    elapsed = time.perf_counter() - start
    final = accuracy(model, x_test, y_test)
    print(f"final test accuracy {final * 100:.2f}%  training time {elapsed:.1f}s")


if __name__ == "__main__":
    main()
