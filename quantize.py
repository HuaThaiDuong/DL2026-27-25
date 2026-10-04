import torch
import torch.ao.quantization as tq
from torch.utils.data import Subset

from common import build_teacher, cifar10, evaluate, get_parser, loader, prepare_int8, setup


def main():
    parser = get_parser(__doc__)
    parser.add_argument("--calib-images", type=int, default=500, help="number of training images for calibration")
    args = parser.parse_args()
    _, ckpt_dir, _ = setup(args)

    model = build_teacher()
    model.load_state_dict(torch.load(ckpt_dir / "resnet18_baseline.pt", map_location="cpu"))
    model = prepare_int8(model)

    with torch.no_grad():
        for images, _ in loader(Subset(cifar10(train=True), range(args.calib_images))):
            model(images)

    model = tq.convert(model)
    file = ckpt_dir / "resnet18_int8.pt"
    torch.save(model.state_dict(), file)
    test_loader = loader(cifar10(train=False, debug=args.debug))
    print(f"INT8 ResNet-18: test accuracy {evaluate(model, test_loader, torch.device('cpu')):.2f}%  -> saved {file}")


if __name__ == "__main__":
    main()
