import torch

from common import build_student, build_teacher, get_loaders, get_parser, setup, train


def main():
    parser = get_parser(__doc__)
    parser.add_argument("--model", choices=["teacher", "student", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1, help="peak learning rate of the one-cycle schedule")
    args = parser.parse_args()
    device, ckpt_dir, _ = setup(args)
    train_loader, test_loader = get_loaders(args.debug)
    epochs = 1 if args.debug else args.epochs

    for name, build, file in [("teacher", build_teacher, "resnet18_baseline.pt"),
                              ("student", build_student, "student_baseline.pt")]:
        if args.model not in (name, "both"):
            continue
        torch.manual_seed(0)
        model = build()
        print(f"\nTraining {name} ({sum(p.numel() for p in model.parameters()):,} parameters), {epochs} epochs")
        acc = train(model, train_loader, test_loader, epochs, args.lr, device)
        torch.save(model.state_dict(), ckpt_dir / file)
        print(f"{name}: final test accuracy {acc:.2f}%  -> saved {ckpt_dir / file}")


if __name__ == "__main__":
    main()
