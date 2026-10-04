# Dataset: CIFAR-10

| | |
|---|---|
| Official URL | https://www.cs.toronto.edu/~kriz/cifar.html |
| Version | CIFAR-10 *python version*: `cifar-10-python.tar.gz`, 170,498,071 bytes, md5 `c58f30108f718f92721af3b95e74349a` |
| Reference | A. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, Technical Report, University of Toronto, 2009 |
| Content | 60,000 RGB images of 32×32 pixels in 10 balanced classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |

This project uses the official dataset unchanged. It creates no new or processed dataset, so there is nothing extra to download.

## Getting the data

Every script downloads the data on first use through `torchvision.datasets.CIFAR10(root="data", download=True)`. That call fetches the archive from the official URL into `data/`, extracts it, and checks the md5 checksum of the archive and of each batch file. To download and check the data on its own:

```bash
python common.py
```

Expected output:

```
train: 50000 images, per-class counts [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
test: 10000 images, per-class counts [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
```

## Data split

The official train/test split, used as is:

| Split | Files | Images | Per class | Used for |
|---|---|---:|---:|---|
| train | `data_batch_1` … `data_batch_5` | 50,000 | 5,000 | training and fine-tuning all models; the **first 500 images** are also the INT8 calibration set |
| test | `test_batch` | 10,000 | 1,000 | every accuracy reported in the README |

There is no separate validation set. All hyperparameters are standard values fixed in advance (they were not tuned on the test set). Each model is saved after its last epoch, so no checkpoint is picked by its test accuracy.

Smoke-test runs (`--debug`) use only the first 1,024 training images and the first 512 test images.

## Preprocessing

Implemented in `cifar10()` in [common.py](common.py):

1. `ToTensor()`: uint8 HWC image → float32 CHW tensor in [0, 1].
2. `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))`: per-channel mean and standard deviation of the 50,000 training images.
3. **Training images only**, as on-the-fly augmentation that is redrawn every epoch, applied before steps 1 and 2:
   - `RandomCrop(32, padding=4)`: zero-pad to 40×40, then take a random 32×32 crop;
   - `RandomHorizontalFlip(p=0.5)`.

Test images and calibration images get steps 1 and 2 only.
