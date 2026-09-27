# PyTorch CIFAR-10 DDP example

Uses the real CIFAR-10 dataset.

- 1 worker: normal PyTorch training.
- 2 or 4 SageMaker workers: PyTorch DistributedDataParallel (DDP).
- Each `ml.g4dn.xlarge` worker contributes one NVIDIA T4 GPU.
- Uses only 4,000 real training images and 1,000 test images to keep runs short.
- Reports elapsed time, throughput and accuracy.
