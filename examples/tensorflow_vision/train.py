import argparse
import json
import os
import time
from pathlib import Path

import tensorflow as tf

def sagemaker_hosts():
    raw = os.getenv("SM_HOSTS")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []

def configure_tf_config():
    hosts = sagemaker_hosts()
    current = os.getenv("SM_CURRENT_HOST")
    if len(hosts) <= 1 or not current:
        return 1, 0

    port = 12345
    workers = [f"{h}:{port}" for h in hosts]
    task_index = hosts.index(current)
    tf_config = {
        "cluster": {"worker": workers},
        "task": {"type": "worker", "index": task_index},
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    return len(hosts), task_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--git-commit", type=str, default="unknown")
    args = ap.parse_args()

    world_size, rank = configure_tf_config()
    strategy = (
        tf.distribute.MultiWorkerMirroredStrategy()
        if world_size > 1
        else tf.distribute.get_strategy()
    )

    data_root = os.getenv("SM_CHANNEL_TRAINING") or os.getenv("LOCAL_DATA_DIR") or "/tmp/cifar10"
    Path(data_root).mkdir(parents=True, exist_ok=True)

    # Real public dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Small real subset to keep the demo cheap and short.
    max_train = min(4000, len(x_train))
    max_test = min(1000, len(x_test))
    x_train, y_train = x_train[:max_train], y_train[:max_train]
    x_test, y_test = x_test[:max_test], y_test[:max_test]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    global_batch = args.batch_size * max(1, world_size)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(max_train, seed=42).batch(global_batch).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(global_batch).prefetch(tf.data.AUTOTUNE)

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    start = time.time()
    hist = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=test_ds,
        verbose=2 if rank == 0 else 0,
    )
    elapsed = time.time() - start

    if rank == 0:
        model_dir = Path(os.getenv("SM_MODEL_DIR", os.getenv("LOCAL_MODEL_DIR", "/tmp/model")))
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_dir / "model.keras")

        samples = max_train * args.epochs
        metrics = {
            "framework": "tensorflow",
            "dataset": "CIFAR-10",
            "workers": world_size,
            "train_samples": max_train,
            "test_samples": max_test,
            "elapsed_seconds": round(elapsed, 3),
            "samples_per_second": round(samples / elapsed, 3) if elapsed else None,
            "accuracy": float(hist.history["accuracy"][-1]),
            "val_accuracy": float(hist.history["val_accuracy"][-1]),
        }
        print("METRICS", json.dumps(metrics))

if __name__ == "__main__":
    main()
