import argparse
import json
import os
import pickle
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def hosts():
    try:
        return json.loads(os.getenv("SM_HOSTS", "[]"))
    except Exception:
        return []


def init_distributed():
    hs = hosts()
    current = os.getenv("SM_CURRENT_HOST")
    if len(hs) <= 1 or not current:
        return False, 0, 1
    rank = hs.index(current)
    os.environ.setdefault("MASTER_ADDR", hs[0])
    os.environ.setdefault("MASTER_PORT", "29500")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=len(hs))
    return True, rank, len(hs)


def locate_and_extract_cifar(channel):
    channel = Path(channel)
    extracted = channel / "cifar-10-batches-py"
    if extracted.exists():
        return extracted

    archives = list(channel.glob("*.tar.gz")) + list(channel.glob("*.tgz"))
    if not archives:
        raise FileNotFoundError("Expected registered CIFAR-10 .tar.gz in SM_CHANNEL_TRAINING")

    with tarfile.open(archives[0], "r:gz") as tar:
        tar.extractall(channel)
    if not extracted.exists():
        raise FileNotFoundError("CIFAR-10 archive extracted but cifar-10-batches-py was not found")
    return extracted


def load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    y = np.asarray(d[b"labels"], dtype="int64")
    return x, y


def load_cifar(channel):
    folder = locate_and_extract_cifar(channel)
    xs, ys = [], []
    for i in range(1, 6):
        x, y = load_batch(folder / f"data_batch_{i}")
        xs.append(x); ys.append(y)
    train_x = np.concatenate(xs); train_y = np.concatenate(ys)
    test_x, test_y = load_batch(folder / "test_batch")
    return train_x, train_y, test_x, test_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--git-commit", type=str, default="unknown")
    args = ap.parse_args()

    distributed, rank, world = init_distributed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    channel = os.getenv("SM_CHANNEL_TRAINING")
    if not channel:
        raise RuntimeError("This reproducible example expects a registered S3 dataset channel")

    train_x, train_y, test_x, test_y = load_cifar(channel)
    train_x, train_y = train_x[:4000], train_y[:4000]
    test_x, test_y = test_x[:1000], test_y[:1000]

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = SmallCNN().to(device)
    if distributed:
        model = DDP(model, device_ids=[0] if torch.cuda.is_available() else None)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    elapsed = time.time() - start

    if rank == 0:
        eval_model = model.module if distributed else model
        eval_model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = eval_model(x).argmax(1)
                correct += int((pred == y).sum())
                total += len(y)

        model_dir = Path(os.getenv("SM_MODEL_DIR", "/tmp/model"))
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(eval_model.state_dict(), model_dir / "model.pt")
        print("METRICS", json.dumps({
            "framework": "pytorch",
            "dataset": "CIFAR-10",
            "workers": world,
            "epochs": args.epochs,
            "elapsed_seconds": round(elapsed, 3),
            "samples_per_second": round((len(train_ds) * args.epochs) / elapsed, 3),
            "accuracy": round(correct / max(total, 1), 4),
        }))

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
