"""Extra analyses on the saved checkpoints. Place this file in report/ and run from the repo root:

    python report/extra_experiments.py            # ~10 min, no training
    python report/extra_experiments.py --control  # + 10-epoch fine-tuning control on the GPU

1. Pruning sensitivity BEFORE fine-tuning (deterministic: global L1 pruning of the saved teacher).
2. INT8 calibration-size ablation (n=500 must reproduce quantize.py's 93.79%: sanity check).
3. Pruning + INT8 combined (pruned checkpoints -> post-training static quantization).
4. Repeated, interleaved CPU latency for all seven checkpoints.
5. --control: fine-tune the UNPRUNED teacher with the pruning budget (10 epochs, lr 0.01), to separate
   the effect of pruning from the effect of extra training.

Writes CSV files to results/extra/. Does not touch results/results.csv or the README.
"""
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.ao.quantization as tq
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Subset

from common import (build_teacher, cifar10, cpu_latency_ms, evaluate, file_size_mb, get_loaders, get_parser,
                    loader, prepare_int8, setup, train)
from evaluate import MODELS

CPU = torch.device("cpu")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def load(path):
    model = build_teacher()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def quantize(model, calib_set, n):
    model = prepare_int8(model)
    with torch.no_grad():
        for images, _ in loader(Subset(calib_set, range(n))):
            model(images)
    return tq.convert(model)


def pruning_before_finetune(ckpt_dir, test_loader, device, amounts):
    rows = []
    for amount in amounts:
        model = load(ckpt_dir / "resnet18_baseline.pt")
        convs = [(m, "weight") for m in model.modules() if isinstance(m, nn.Conv2d)]
        if amount > 0:
            prune.global_unstructured(convs, pruning_method=prune.L1Unstructured, amount=amount)
        rows.append(dict(conv_sparsity=amount, accuracy_before_finetune=round(evaluate(model, test_loader, device), 2)))
        print(rows[-1])
    return rows


def calibration_size(ckpt_dir, test_loader, calib_set, sizes):
    rows = []
    for n in sizes:
        model = quantize(load(ckpt_dir / "resnet18_baseline.pt"), calib_set, n)
        rows.append(dict(calibration_images=n, int8_accuracy=round(evaluate(model, test_loader, CPU), 2)))
        print(rows[-1])
    return rows


def prune_then_quantize(ckpt_dir, test_loader, calib_set, n=500):
    rows = []
    for amount in (30, 50, 70):
        src = ckpt_dir / f"resnet18_pruned{amount}.pt"
        if not src.exists():
            print(f"skipping {src.name}: not found")
            continue
        model = quantize(load(src), calib_set, n)
        dst = ckpt_dir / f"resnet18_pruned{amount}_int8.pt"
        torch.save(model.state_dict(), dst)
        size, gz = file_size_mb(dst)
        rows.append(dict(model=f"pruned {amount}% + INT8", accuracy=round(evaluate(model, test_loader, CPU), 2),
                         size_mib=round(size, 3), gzip_mib=round(gz, 3), latency_ms=round(cpu_latency_ms(model), 3)))
        print(rows[-1])
    return rows


def finetune_control(ckpt_dir, device, debug):
    train_loader, test_loader = get_loaders(debug)
    torch.manual_seed(0)
    model = load(ckpt_dir / "resnet18_baseline.pt")
    acc = train(model, train_loader, test_loader, 1 if debug else 10, 0.01, device)
    torch.save(model.state_dict(), ckpt_dir / "resnet18_finetune_control.pt")
    return [dict(model="ResNet-18 + 10 epochs fine-tuning, no pruning", accuracy=round(acc, 2))]


def repeated_latency(ckpt_dir, rounds):
    models = {}
    for name, file, build, *_ in MODELS:
        if (ckpt_dir / file).exists():
            model = build()
            model.load_state_dict(torch.load(ckpt_dir / file, map_location="cpu"))
            models[name] = model.eval()
    names, samples = list(models), {name: [] for name in models}
    for r in range(rounds):  # alternate the order so no model is always measured first or last
        for name in (names if r % 2 == 0 else names[::-1]):
            samples[name].append(cpu_latency_ms(models[name]))
    return [dict(model=n, threads=torch.get_num_threads(), rounds=rounds, median_ms=round(statistics.median(s), 3),
                 min_ms=round(min(s), 3), max_ms=round(max(s), 3)) for n, s in samples.items()]


def main():
    parser = get_parser(__doc__)
    parser.add_argument("--amounts", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--calib-sizes", type=int, nargs="+", default=[32, 128, 500, 2000])
    parser.add_argument("--rounds", type=int, default=5, help="interleaved timing rounds per model")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for timing (0 = PyTorch default)")
    parser.add_argument("--control", action="store_true", help="also run the equal-budget fine-tuning control")
    args = parser.parse_args()
    device, ckpt_dir, results_dir = setup(args)
    out = results_dir / "extra"
    out.mkdir(parents=True, exist_ok=True)
    test_loader = loader(cifar10(train=False, debug=args.debug))
    calib_set = cifar10(train=True)

    write_csv(out / "pruning_before_finetune.csv", pruning_before_finetune(ckpt_dir, test_loader, device, args.amounts))
    write_csv(out / "int8_calibration_size.csv", calibration_size(ckpt_dir, test_loader, calib_set, args.calib_sizes))
    write_csv(out / "prune_then_int8.csv", prune_then_quantize(ckpt_dir, test_loader, calib_set))
    if args.control:
        write_csv(out / "finetune_control.csv", finetune_control(ckpt_dir, device, args.debug))
    if args.threads:
        torch.set_num_threads(args.threads)
    write_csv(out / "latency_repeated.csv", repeated_latency(ckpt_dir, args.rounds))


if __name__ == "__main__":
    main()
