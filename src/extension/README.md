# Extension: CIFAR-100

Extends the CIFAR-10 ResNet-18 experiment from [`../replication`](../replication) to CIFAR-100, to test whether the Gaussianity of InfoNCE representations still holds with 10x more classes. Architecture, augmentation, and hyperparameters are identical to the replication; only the dataset changes.

## 📦 Contents

- `cifar100_resnet18/train_contrastive.py`: CIFAR-100 contrastive (InfoNCE) training with final Gaussianity diagnostics on the backbone and projector spaces.
- `cifar100_resnet18/train_supervised.py`: matched CIFAR-100 supervised baseline (100-way linear head) with final Gaussianity diagnostics on the backbone space.
- `cifar100_resnet18/data.py`: SimCLR augmentation with CIFAR-100 normalization statistics.
- `cifar100_resnet18/model.py`, `cifar100_resnet18/diagnostics.py`: unchanged from the replication.

## 🚀 Running

```bash
cd cifar100_resnet18
python train_contrastive.py
python train_supervised.py
```

The dataset downloads to `<repo>/data` on first run, regardless of the working directory.
