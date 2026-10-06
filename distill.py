"""Knowledge distillation from the ResNet-18 teacher into the compact student CNN."""
import torch
import torch.nn.functional as F

from common import build_student, build_teacher, get_loaders, get_parser, setup, train


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    kl = F.kl_div(F.log_softmax(student_logits / T, dim=1), F.log_softmax(teacher_logits / T, dim=1),
                  reduction="batchmean", log_target=True)
    return (1 - alpha) * F.cross_entropy(student_logits, labels) + alpha * T * T * kl


def main():
    parser = get_parser(__doc__)
    parser.add_argument("--epochs", type=int, default=30, help="keep equal to the student baseline for a fair comparison")
    parser.add_argument("--lr", type=float, default=0.1, help="peak learning rate of the one-cycle schedule")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of the distillation (KL) term")
    args = parser.parse_args()
    device, ckpt_dir, _ = setup(args)
    train_loader, test_loader = get_loaders(args.debug)
    epochs = 1 if args.debug else args.epochs

    teacher = build_teacher()
    teacher.load_state_dict(torch.load(ckpt_dir / "resnet18_baseline.pt", map_location="cpu"))
    teacher.to(device).eval().requires_grad_(False)

    def loss_fn(student_logits, images, labels):
        with torch.no_grad():
            teacher_logits = teacher(images)
        return distillation_loss(student_logits, teacher_logits, labels, args.temperature, args.alpha)

    torch.manual_seed(0)
    student = build_student()
    print(f"\nDistilling into the student (T={args.temperature}, alpha={args.alpha}), {epochs} epochs")
    acc = train(student, train_loader, test_loader, epochs, args.lr, device, loss_fn)
    file = ckpt_dir / "student_distilled.pt"
    torch.save(student.state_dict(), file)
    print(f"distilled student: final test accuracy {acc:.2f}%  -> saved {file}")


if __name__ == "__main__":
    main()
