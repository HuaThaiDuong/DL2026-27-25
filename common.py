import argparse
import gzip
import statistics
import time
import warnings
from collections import Counter
from pathlib import Path

import torch
import torch.ao.quantization as tq
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
from torchvision.models.quantization import resnet18

warnings.filterwarnings(
    "ignore",
    message="(torch.ao.quantization is deprecated|Please use quant_min|torch.quantize_per_tensor"
            "|TypedStorage is deprecated|must run observer)",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
BATCH_SIZE = 128


def get_parser(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debug", action="store_true",
                        help="quick smoke test: 1024 train / 512 test images, 1 epoch; "
                             "outputs go to checkpoints/debug and results/debug")
    return parser


def setup(args):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    sub = "debug" if args.debug else ""
    ckpt_dir, results_dir = ROOT / "checkpoints" / sub, ROOT / "results" / sub
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (" (DEBUG run)" if args.debug else ""))
    return device, ckpt_dir, results_dir


def cifar10(train, augment=False, debug=False):
    tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []
    dataset = datasets.CIFAR10(DATA_DIR, train=train, download=True,
                               transform=T.Compose(tf + [T.ToTensor(), T.Normalize(MEAN, STD)]))
    return Subset(dataset, range(1024 if train else 512)) if debug else dataset


def loader(dataset, shuffle=False):
    return DataLoader(dataset, BATCH_SIZE, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def get_loaders(debug=False):
    return loader(cifar10(True, augment=True, debug=debug), shuffle=True), loader(cifar10(False, debug=debug))


def build_teacher():
    model = resnet18(weights=None, quantize=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_student():
    def block(c_in, c_out):
        return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    return nn.Sequential(*block(3, 64), *block(64, 128), *block(128, 256),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10))


def prepare_int8(model):
    model.eval().fuse_model()
    model.qconfig = tq.get_default_qconfig("x86")
    return tq.prepare(model)


def train(model, train_loader, test_loader, epochs, lr, device, loss_fn=None):
    loss_fn = loss_fn or (lambda logits, images, labels: F.cross_entropy(logits, labels))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(train_loader))
    for epoch in range(1, epochs + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_fn(model(images), images, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.detach()
        acc = evaluate(model, test_loader, device)
        print(f"  epoch {epoch:3d}/{epochs}  loss {total_loss.item() / len(train_loader):.4f}  "
              f"test acc {acc:6.2f}%  ({time.time() - start:.0f}s)")
    return acc


def evaluate(model, loader, device):
    model.to(device).eval()
    with torch.inference_mode():
        correct = sum((model(x.to(device)).argmax(1).cpu() == y).sum().item() for x, y in loader)
    return 100.0 * correct / len(loader.dataset)


def file_size_mb(path):
    data = Path(path).read_bytes()
    return len(data) / 2**20, len(gzip.compress(data, compresslevel=6)) / 2**20


def count_params(model):
    tensors = []
    for m in model.modules():
        if callable(getattr(m, "weight", None)):
            tensors += [m.weight().dequantize(), m.bias()]
        else:
            tensors += m.parameters(recurse=False)
    tensors = [t for t in tensors if t is not None]
    return sum(t.numel() for t in tensors), sum(int(t.count_nonzero()) for t in tensors)


def cpu_latency_ms(model, runs=100, warmup=10):
    model.cpu().eval()
    x = torch.randn(1, 3, 32, 32)
    times = []
    with torch.inference_mode():
        for _ in range(warmup + runs):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)
    return statistics.median(times[warmup:]) * 1000


if __name__ == "__main__":
    for split in ("train", "test"):
        ds = cifar10(train=split == "train")
        print(f"{split}: {len(ds)} images, per-class counts {sorted(Counter(ds.targets).values())}")
