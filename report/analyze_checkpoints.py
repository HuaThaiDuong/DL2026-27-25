"""Per-image prediction audit of the saved checkpoints (teacher, INT8, student CE, student KD) on the CIFAR-10 test set.

Read-only on checkpoints/, CPU only, 4 threads. Run from the repo root:

    python report/analyze_checkpoints.py

Writes predictions.csv, summary.json, confusion_matrices.png and examples.png to report/analysis/.
Does not touch results/ or the README.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from matplotlib.figure import Figure

from common import MEAN, ROOT, STD, cifar10, loader
from evaluate import MODELS

CKPT_DIR = ROOT / "checkpoints"
OUT = Path(__file__).resolve().parent / "analysis"
AUDITED = {"teacher": "ResNet-18 baseline", "int8": "ResNet-18 INT8 (PTQ)",
           "student_ce": "Student CNN baseline", "student_kd": "Student CNN distilled"}


def load(name):
    _, file, build, *_ = next(m for m in MODELS if m[0] == name)
    path = CKPT_DIR / file
    if not path.exists():
        raise SystemExit(f"{path} not found: download the checkpoints first (see README).")
    model = build()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval(), path


def predict(model, test_loader):
    with torch.inference_mode():
        return torch.cat([model(x).argmax(1) for x, _ in test_loader])


def confusion(labels, preds):
    return torch.bincount(labels * 10 + preds, minlength=100).view(10, 10)


def plot_confusions(matrices, classes, path):
    short = ["air", "auto"] + classes[2:]
    fig = Figure(figsize=(12, 5.6), dpi=150, layout="constrained")
    for ax, (title, cm) in zip(fig.subplots(1, len(matrices)), matrices.items()):
        pct = 100 * cm / cm.sum(1, keepdim=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(10):
            for j in range(10):
                v = pct[i, j].item()
                if v > 0:
                    ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center",
                            fontsize=7, color="white" if v > 50 else "black")
        ax.set_xticks(range(10), short, rotation=45, fontsize=8)
        ax.set_yticks(range(10), short, fontsize=8)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("true", fontsize=9)
        ax.set_title(title, fontsize=11)
    fig.savefig(path)


def plot_examples(groups, test_set, preds, classes, path):
    fig = Figure(figsize=(9, 8.5), dpi=150, layout="constrained")
    axes = fig.subplots(2, 3).flatten()
    cases = [(title, i) for title, idx in groups.items() for i in idx]
    for ax, (title, i) in zip(axes, cases):
        image, label = test_set[i]
        pixels = (image * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)
        ax.imshow(pixels.permute(1, 2, 0))
        p = {k: classes[v[i]] for k, v in preds.items()}
        ax.set_title(f"#{i} | {title}", fontsize=9, loc="left", fontweight="bold")
        ax.set_xlabel(f"True: {classes[label]}\nT: {p['teacher']} | CE: {p['student_ce']}\nKD: {p['student_kd']}",
                      fontsize=8, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(cases):]:
        ax.axis("off")
    fig.savefig(path)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    test_set = cifar10(train=False)
    classes = test_set.classes
    test_loader = loader(test_set)
    labels = torch.tensor(test_set.targets)

    preds, hashes = {}, {}
    for key, name in AUDITED.items():
        model, path = load(name)
        hashes[key] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        preds[key] = predict(model, test_loader)
        print(f"{name}: {100 * (preds[key] == labels).float().mean():.2f}%")

    with open(OUT / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true"] + list(preds))
        for i, y in enumerate(labels.tolist()):
            writer.writerow([i, classes[y]] + [classes[p[i]] for p in preds.values()])

    ok = {k: p == labels for k, p in preds.items()}
    matrices = {k: confusion(labels, p) for k, p in preds.items()}
    confusions = {}
    for k, cm in matrices.items():
        off = cm.clone().fill_diagonal_(0).flatten()
        top = off.topk(5)
        confusions[k] = [{"true": classes[i // 10], "predicted": classes[i % 10], "count": c}
                         for c, i in zip(top.values.tolist(), top.indices.tolist())]

    summary = {
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "test_images": len(labels),
        "checkpoints": hashes,
        "accuracy": {k: round(100 * v.float().mean().item(), 2) for k, v in ok.items()},
        "per_class_recall": {k: {c: round(100 * cm[i, i].item() / cm[i].sum().item(), 2) for i, c in enumerate(classes)}
                             for k, cm in matrices.items()},
        "largest_confusions": confusions,
        "kd_vs_ce": {"kd_corrects_ce_error": int((~ok["student_ce"] & ok["student_kd"]).sum()),
                     "kd_introduces_new_error": int((ok["student_ce"] & ~ok["student_kd"]).sum()),
                     "both_wrong": int((~ok["student_ce"] & ~ok["student_kd"]).sum())},
        "int8_vs_teacher": {"int8_corrects_teacher_error": int((~ok["teacher"] & ok["int8"]).sum()),
                            "int8_loses_teacher_correct": int((ok["teacher"] & ~ok["int8"]).sum()),
                            "disagreements": int((preds["teacher"] != preds["int8"]).sum())},
    }
    groups = {"KD correction": (~ok["student_ce"] & ok["student_kd"]).nonzero().flatten()[:2].tolist(),
              "KD regression": (ok["student_ce"] & ~ok["student_kd"]).nonzero().flatten()[:2].tolist(),
              "Shared failure": (~ok["teacher"] & ~ok["student_ce"] & ~ok["student_kd"]).nonzero().flatten()[:2].tolist()}
    summary["examples"] = groups
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_confusions({"ResNet-18 teacher": matrices["teacher"], "Distilled student": matrices["student_kd"]},
                    classes, OUT / "confusion_matrices.png")
    plot_examples(groups, test_set, preds, classes, OUT / "examples.png")
    print(f"wrote predictions.csv, summary.json, confusion_matrices.png and examples.png to {OUT}")


if __name__ == "__main__":
    main()
