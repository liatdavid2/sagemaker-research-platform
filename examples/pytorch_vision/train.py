import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

def sm_hosts():
    raw = os.getenv("SM_HOSTS")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []

def init_distributed():
    hosts = sm_hosts()
    current = os.getenv("SM_CURRENT_HOST")

    if len(hosts) <= 1 or not current:
        return False, 0, 1

    rank = hosts.index(current)
    world_size = len(hosts)
    master = hosts[0]

    os.environ.setdefault("MASTER_ADDR", master)
    os.environ.setdefault("MASTER_PORT", "29500")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master}:29500",
        rank=rank,
        world_size=world_size,
    )
    return True, rank, world_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--git-commit", type=str, default="unknown")
    args = ap.parse_args()

    distributed, rank, world_size = init_distributed()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    data_root = os.getenv("SM_CHANNEL_TRAINING") or os.getenv("LOCAL_DATA_DIR") or "/tmp/cifar10"
    Path(data_root).mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # Real subsets to keep the demonstration inexpensive.
    train_ds = Subset(train_ds, range(min(4000, len(train_ds))))
    test_ds = Subset(test_ds, range(min(1000, len(test_ds))))

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SmallCNN().to(device)
    if distributed:
        model = DDP(model, device_ids=[0] if torch.cuda.is_available() else None)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    model.train()
    total_seen = 0
    total_loss = 0.0

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_seen += y.size(0)
            total_loss += float(loss.item()) * y.size(0)

    elapsed = time.time() - start

    # Only rank 0 evaluates and writes artifacts.
    if rank == 0:
        eval_model = model.module if distributed else model
        eval_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = eval_model(x).argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += y.size(0)

        model_dir = Path(os.getenv("SM_MODEL_DIR", os.getenv("LOCAL_MODEL_DIR", "/tmp/model")))
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(eval_model.state_dict(), model_dir / "model.pt")

        global_samples = len(train_ds) * args.epochs
        metrics = {
            "framework": "pytorch",
            "dataset": "CIFAR-10",
            "workers": world_size,
            "device": str(device),
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "elapsed_seconds": round(elapsed, 3),
            "samples_per_second": round(global_samples / elapsed, 3) if elapsed else None,
            "accuracy": round(correct / max(total, 1), 4),
            "average_loss_rank0": round(total_loss / max(total_seen, 1), 5),
        }
        print("METRICS", json.dumps(metrics))

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
