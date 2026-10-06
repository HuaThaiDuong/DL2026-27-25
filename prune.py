"""Global L1 unstructured pruning of the teacher's conv weights, followed by fine-tuning."""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from common import build_teacher, evaluate, get_loaders, get_parser, setup, train


def main():
    parser = get_parser(__doc__)
    parser.add_argument("--amounts", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                        help="fractions of conv weights to remove")
    parser.add_argument("--epochs", type=int, default=10, help="fine-tuning epochs per sparsity level")
    parser.add_argument("--lr", type=float, default=0.01, help="peak fine-tuning learning rate")
    args = parser.parse_args()
    device, ckpt_dir, _ = setup(args)
    train_loader, test_loader = get_loaders(args.debug)
    epochs = 1 if args.debug else args.epochs

    for amount in args.amounts:
        model = build_teacher()
        model.load_state_dict(torch.load(ckpt_dir / "resnet18_baseline.pt", map_location="cpu"))
        convs = [(m, "weight") for m in model.modules() if isinstance(m, nn.Conv2d)]

        prune.global_unstructured(convs, pruning_method=prune.L1Unstructured, amount=amount)
        print(f"\nPruned {amount:.0%} of conv weights: test acc {evaluate(model, test_loader, device):.2f}% "
              f"before fine-tuning. Fine-tuning {epochs} epochs")
        acc = train(model, train_loader, test_loader, epochs, args.lr, device)

        for module, name in convs:
            prune.remove(module, name)
        zeros = sum(int((m.weight == 0).sum()) for m, _ in convs)
        total = sum(m.weight.numel() for m, _ in convs)
        file = ckpt_dir / f"resnet18_pruned{round(amount * 100)}.pt"
        torch.save(model.state_dict(), file)
        print(f"conv sparsity {zeros / total:.1%}, final test accuracy {acc:.2f}%  -> saved {file}")


if __name__ == "__main__":
    main()
