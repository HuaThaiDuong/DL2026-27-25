# Efficient Deep Learning on CIFAR-10: Pruning, Quantization, Distillation

Deep Learning final project 2026–2027, **Group 27, Project 25**.
We train a ResNet-18 on CIFAR-10 and compress it in three ways: global L1 pruning (30/50/70 %) + fine-tuning, post-training INT8 quantization, and knowledge distillation into a 0.37M-parameter CNN. All models are compared on accuracy, size and CPU latency.

## Team members

| # | Student ID | Full name             | Email                         |
| - | ---------- | --------------------- | ----------------------------- |
| 1 | 23BI14123  | Hứa Thái Dương        | duonght.23bi14123@usth.edu.vn |
| 2 | 23BI14160  | Nguyễn Ngọc Minh Hiếu | hieunnm.23bi14160@usth.edu.vn |
| 3 | 23BA14271  | Dương Đức Thịnh       | thinhdd.23ba14271@usth.edu.vn |
| 4 | 22BA13225  | Hồ Nguyễn Hoàng Nam   | namhnh.22ba13225@usth.edu.vn  |
| 5 | 23BA14272  | Nguyễn Ngọc Thịnh     | thinhnn.23ba14272@usth.edu.vn |
| 6 | 22BA13101  | Nguyễn Thái Duy       | duynt.22ba13101@usth.edu.vn   |
| 7 | 22BA13064  | Hoàng Trí Đạt         | datht.22ba13064@usth.edu.vn   |

## Files

- [common.py](common.py): data loading, models, training loop (`python common.py` downloads CIFAR-10)
- [train_baselines.py](train_baselines.py), [prune.py](prune.py), [quantize.py](quantize.py), [distill.py](distill.py): training and compression
- [evaluate.py](evaluate.py): measures every model, writes `results/` and the table below
- [script_demo.py](script_demo.py): inference demo on random test images
- [report/](report/): extra analyses used in the report
- [DATA.md](DATA.md): dataset version, split and preprocessing. Dataset: [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html), downloaded automatically

## Installation

Python 3.13, an NVIDIA GPU is recommended for training.

```
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Without an NVIDIA GPU, remove the `--extra-index-url` line in `requirements.txt` first.

## Reproducing the results

Full pipeline (about 40 min on an RTX 4050 laptop GPU):

```
python common.py
python train_baselines.py
python prune.py
python quantize.py
python distill.py
python evaluate.py
python script_demo.py --seed 0
```

To skip training, download the seven `.pt` files from the [v1.0 release](https://github.com/HuaThaiDuong/DL2026-27-25/releases/tag/v1.0) into `checkpoints/` and run only the last two commands.
Accuracy and sizes are reproducible; CPU latency depends on the machine. `--debug` runs a 1-epoch smoke test on a small subset.

## Results

<!-- results:start -->
| Model | Accuracy (%) | Δ acc (pp) | Size (MiB) | gzip (MiB) | Compression | Params | Non-zero | CPU latency (ms) | Speed-up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 baseline | 93.77 | +0.00 | 42.70 | 39.59 | 1.0× | 11.17M | 11.17M | 5.79 | 1.0× |
| ResNet-18 pruned 30% | 94.14 | +0.37 | 42.70 | 30.80 | 1.3× | 11.17M | 7.83M | 5.65 | 1.0× |
| ResNet-18 pruned 50% | 93.92 | +0.15 | 42.70 | 23.69 | 1.7× | 11.17M | 5.59M | 5.47 | 1.1× |
| ResNet-18 pruned 70% | 93.87 | +0.10 | 42.70 | 15.91 | 2.5× | 11.17M | 3.36M | 5.83 | 1.0× |
| ResNet-18 INT8 (PTQ) | 93.79 | +0.02 | 10.78 | 9.44 | 4.2× | 11.17M | 11.02M | 2.46 | 2.4× |
| Student CNN baseline | 86.16 | -7.61 | 1.44 | 1.33 | 29.7× | 0.37M | 0.37M | 0.86 | 6.8× |
| Student CNN distilled | 87.75 | -6.02 | 1.44 | 1.33 | 29.8× | 0.37M | 0.37M | 0.54 | 10.6× |

CPU latency = one 32×32 image (batch size 1), median of 100 runs after 10 warm-up runs, 14 threads, PyTorch 2.14.1+cu132. MiB = 2^20 bytes. Compression, speed-up and Δ acc are relative to the ResNet-18 baseline; compression uses the gzip size, because pruned weights are only smaller on disk once their zeros are compressed.

- **Smallest model within 0.5 pp of the baseline:** ResNet-18 INT8 (PTQ): 9.44 MiB gzip (4.2× smaller), +0.02 pp accuracy.
- **Fastest CPU inference:** Student CNN distilled: 0.54 ms per image, 10.6× faster than the baseline, -6.02 pp accuracy.
- **Distillation gain:** +1.59 pp over the same CNN trained with cross-entropy only, with identical architecture and size.
<!-- results:end -->

![Accuracy vs. model size](results/accuracy_vs_size.png)
