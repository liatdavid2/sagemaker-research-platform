import argparse
import json
import os
import pickle
import tarfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


def configure_workers():
    try:
        hosts = json.loads(os.getenv("SM_HOSTS", "[]"))
    except Exception:
        hosts = []
    current = os.getenv("SM_CURRENT_HOST")
    if len(hosts) <= 1 or not current:
        return 1, 0
    workers = [f"{h}:12345" for h in hosts]
    rank = hosts.index(current)
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {"worker": workers},
        "task": {"type": "worker", "index": rank},
    })
    return len(hosts), rank


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
    return extracted


def load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32") / 255.0
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

    world, rank = configure_workers()
    strategy = tf.distribute.MultiWorkerMirroredStrategy() if world > 1 else tf.distribute.get_strategy()

    channel = os.getenv("SM_CHANNEL_TRAINING")
    if not channel:
        raise RuntimeError("This reproducible example expects a registered S3 dataset channel")

    x_train, y_train, x_test, y_test = load_cifar(channel)
    x_train, y_train = x_train[:4000], y_train[:4000]
    x_test, y_test = x_test[:1000], y_test[:1000]

    global_batch = args.batch_size * world
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(4000, seed=42).batch(global_batch).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(global_batch).prefetch(tf.data.AUTOTUNE)

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Input((32,32,3)),
            tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64,3,padding="same",activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation="relu"),
            tf.keras.layers.Dense(10,activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    start = time.time()
    hist = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, verbose=2 if rank == 0 else 0)
    elapsed = time.time() - start

    if rank == 0:
        model_dir = Path(os.getenv("SM_MODEL_DIR", "/tmp/model"))
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_dir / "model.keras")
        print("METRICS", json.dumps({
            "framework": "tensorflow",
            "dataset": "CIFAR-10",
            "workers": world,
            "epochs": args.epochs,
            "elapsed_seconds": round(elapsed, 3),
            "samples_per_second": round((len(x_train) * args.epochs) / elapsed, 3),
            "accuracy": float(hist.history["accuracy"][-1]),
            "val_accuracy": float(hist.history["val_accuracy"][-1]),
        }))


if __name__ == "__main__":
    main()
