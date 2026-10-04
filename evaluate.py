import csv

import torch
import torch.ao.quantization as tq
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from common import (ROOT, build_student, build_teacher, cifar10, count_params, cpu_latency_ms, evaluate,
                    file_size_mb, get_parser, loader, prepare_int8, setup)


def build_int8():
    return tq.convert(prepare_int8(build_teacher()))


MODELS = [
    ("ResNet-18 baseline", "resnet18_baseline.pt", build_teacher, "Baseline", "o"),
    ("ResNet-18 pruned 30%", "resnet18_pruned30.pt", build_teacher, "Pruning", "^"),
    ("ResNet-18 pruned 50%", "resnet18_pruned50.pt", build_teacher, "Pruning", "s"),
    ("ResNet-18 pruned 70%", "resnet18_pruned70.pt", build_teacher, "Pruning", "D"),
    ("ResNet-18 INT8 (PTQ)", "resnet18_int8.pt", build_int8, "Quantization", "o"),
    ("Student CNN baseline", "student_baseline.pt", build_student, "Baseline", "s"),
    ("Student CNN distilled", "student_distilled.pt", build_student, "Distillation", "s"),
]
MARKERS = {name: marker for name, *_, marker in MODELS}

COLORS = {"Baseline": "#52514e", "Pruning": "#2a78d6", "Quantization": "#eb6834", "Distillation": "#1baf7a"}
INK, INK_2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def markdown_report(rows):
    base = rows[0]
    lines = ["| Model | Accuracy (%) | Δ acc (pp) | Size (MB) | gzip (MB) | Compression | Params | Non-zero "
             "| CPU latency (ms) | Speed-up |", "|---|" + "---:|" * 9]
    for r in rows:
        lines.append(f"| {r['model']} | {r['accuracy']:.2f} | {r['acc_change_pp']:+.2f} | {r['size_mb']:.2f} "
                     f"| {r['gzip_mb']:.2f} | {r['compression_x']:.1f}× | {r['params'] / 1e6:.2f}M "
                     f"| {r['nonzero_params'] / 1e6:.2f}M | {r['latency_ms']:.2f} | {r['speedup_x']:.1f}× |")
    lines += ["", "CPU latency = one 32×32 image (batch size 1), median of 100 runs after 10 warm-up runs, "
                  f"{torch.get_num_threads()} threads, PyTorch {torch.__version__}. MB = 2^20 bytes. "
                  "Compression, speed-up and Δ acc are relative to the ResNet-18 baseline; compression uses the "
                  "gzip size, because pruned weights are only smaller on disk once their zeros are compressed.", ""]

    methods = [r for r in rows if r["technique"] != "Baseline"]
    smaller = [r for r in methods if r["gzip_mb"] < base["gzip_mb"]]
    if smaller:
        best = max(smaller, key=lambda r: r["acc_change_pp"] / (base["gzip_mb"] - r["gzip_mb"]))
        lines.append(f"- **Best accuracy retention per MB saved:** {best['model']}: {best['acc_change_pp']:+.2f} pp "
                     f"accuracy for {base['gzip_mb'] - best['gzip_mb']:.1f} MB saved ({best['compression_x']:.1f}× smaller).")
    if methods:
        fast = min(methods, key=lambda r: r["latency_ms"])
        lines.append(f"- **Fastest CPU inference:** {fast['model']}: {fast['latency_ms']:.2f} ms per image, "
                     f"{fast['speedup_x']:.1f}× faster than the baseline, {fast['acc_change_pp']:+.2f} pp accuracy.")
    students = {r["model"]: r for r in rows if r["model"].startswith("Student")}
    if len(students) == 2:
        gain = students["Student CNN distilled"]["accuracy"] - students["Student CNN baseline"]["accuracy"]
        lines.append(f"- **Distillation gain:** {gain:+.2f} pp over the same CNN trained with cross-entropy only, "
                     "at identical size and latency.")
    return "\n".join(lines)


def scatter_plot(rows, x_key, x_label, title, path):
    fig = Figure(figsize=(7, 4.6), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.add_subplot(facecolor=SURFACE)
    for r in rows:
        color, marker = COLORS[r["technique"]], MARKERS[r["model"]]
        if r["technique"] == "Baseline":
            ax.scatter(r[x_key], r["accuracy"], s=70, marker=marker, facecolors="none", edgecolors=color,
                       linewidths=1.8, label=r["model"], zorder=4)
        else:
            ax.scatter(r[x_key], r["accuracy"], s=70, marker=marker, color=color, edgecolors=SURFACE,
                       linewidths=1.5, label=r["model"], zorder=3)
    ax.set_xscale("log")
    ax.margins(x=0.15, y=0.12)
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_2, labelsize=8)
    ax.set_xlabel(x_label, color=INK_2, fontsize=9)
    ax.set_ylabel("CIFAR-10 test accuracy (%)", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=8, loc="best")
    fig.savefig(path)


def update_readme(report):
    readme = ROOT / "README.md"
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    text = readme.read_text(encoding="utf-8")
    if start in text and end in text:
        head, rest = text.split(start, 1)
        readme.write_text(f"{head}{start}\n{report}\n{end}{rest.split(end, 1)[1]}", encoding="utf-8")
        print("updated the results section of README.md")


def main():
    args = get_parser(__doc__).parse_args()
    device, ckpt_dir, results_dir = setup(args)
    if not (ckpt_dir / MODELS[0][1]).exists():
        raise SystemExit(f"{ckpt_dir / MODELS[0][1]} not found: run train_baselines.py first.")
    test_loader = loader(cifar10(train=False, debug=args.debug))
    if device.type == "cuda":
        torch.zeros(1, device=device)

    rows = []
    for name, file, build, technique, _ in MODELS:
        path = ckpt_dir / file
        if not path.exists():
            print(f"skipping {name}: {path.name} not found")
            continue
        model = build()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        latency = cpu_latency_ms(model)
        size, gzip_size = file_size_mb(path)
        params, nonzero = count_params(model)
        acc = evaluate(model, test_loader, torch.device("cpu") if build is build_int8 else device)
        rows.append(dict(model=name, technique=technique, accuracy=round(acc, 2), size_mb=round(size, 3),
                         gzip_mb=round(gzip_size, 3), params=params, nonzero_params=nonzero,
                         latency_ms=round(latency, 3)))
        print(rows[-1])

    base = rows[0]
    for r in rows:
        r["acc_change_pp"] = round(r["accuracy"] - base["accuracy"], 2)
        r["compression_x"] = round(base["gzip_mb"] / r["gzip_mb"], 2)
        r["speedup_x"] = round(base["latency_ms"] / r["latency_ms"], 2)

    with open(results_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = markdown_report(rows)
    (results_dir / "results.md").write_text(report + "\n", encoding="utf-8")
    scatter_plot(rows, "gzip_mb", "Model size (MB, gzip-compressed checkpoint, log scale)",
                 "Accuracy vs. model size", results_dir / "accuracy_vs_size.png")
    scatter_plot(rows, "latency_ms", "CPU inference latency (ms per image, batch size 1, log scale)",
                 "Accuracy vs. CPU latency", results_dir / "accuracy_vs_latency.png")
    print(f"wrote results.csv, results.md and two plots to {results_dir}")
    if not args.debug:
        update_readme(report)


if __name__ == "__main__":
    main()
