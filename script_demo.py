import random

import torch
from matplotlib.figure import Figure

from common import MEAN, STD, cifar10, get_parser, setup
from evaluate import MODELS


def load_models(ckpt_dir):
    models = {}
    for name, file, build, *_ in MODELS:
        path = ckpt_dir / file
        if not path.exists():
            print(f"skipping {name}: {path.name} not found")
            continue
        model = build()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        models[name] = model.eval()
    return models


def predict(model, images):
    with torch.inference_mode():
        return model(images).softmax(1).max(1)


def save_figure(images, labels, n_correct, n_models, classes, path):
    fig = Figure(figsize=(2.2 * len(images), 2.8), dpi=120, layout="constrained")
    pixels = (images * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)
    for i, ax in enumerate(fig.subplots(1, len(images), squeeze=False)[0]):
        ax.imshow(pixels[i].permute(1, 2, 0))
        ax.set_title(f"[{i + 1}] {classes[labels[i]]}", fontsize=10)
        ax.set_xlabel(f"{n_correct[i]}/{n_models} models correct", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path)


def main():
    parser = get_parser("Predict a few random CIFAR-10 test images with every saved model.")
    parser.add_argument("-n", type=int, default=5, help="number of images (default 5)")
    parser.add_argument("--seed", type=int, help="fix the image choice (default: new images every run)")
    args = parser.parse_args()
    _, ckpt_dir, results_dir = setup(args)
    models = load_models(ckpt_dir)
    if not models:
        raise SystemExit(f"no checkpoints in {ckpt_dir}")

    test_set = cifar10(train=False)
    classes = test_set.classes
    idx = random.Random(args.seed).sample(range(len(test_set)), args.n)
    images = torch.stack([test_set[i][0] for i in idx])
    labels = [test_set[i][1] for i in idx]

    print(f"\n{'image':24}" + "".join(f"{f'[{i + 1}] #{j}':19}" for i, j in enumerate(idx)))
    print(f"{'true label':24}" + "".join(f"{classes[y]:19}" for y in labels))
    n_correct = [0] * args.n
    for name, model in models.items():
        conf, pred = predict(model, images)
        cells = []
        for i, (c, p) in enumerate(zip(conf.tolist(), pred.tolist())):
            n_correct[i] += p == labels[i]
            cells.append(f"{classes[p]} {c:.0%}" + ("" if p == labels[i] else " *"))
        print(f"{name:24}" + "".join(f"{cell:19}" for cell in cells))
    print("* = wrong prediction")

    save_figure(images, labels, n_correct, len(models), classes, results_dir / "demo.png")
    print(f"saved {results_dir / 'demo.png'}")


if __name__ == "__main__":
    main()
