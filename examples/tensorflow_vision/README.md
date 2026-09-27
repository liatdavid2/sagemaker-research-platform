# TensorFlow CIFAR-10 GPU example

Uses the real CIFAR-10 dataset.

- 1 worker: normal TensorFlow/Keras training.
- 2 or 4 SageMaker workers: `MultiWorkerMirroredStrategy`.
- Uses only 4,000 real training images and 1,000 test images to keep runs short and inexpensive.
- Reports elapsed time, throughput and accuracy.
