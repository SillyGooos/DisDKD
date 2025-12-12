from pathlib import Path
import torch
from utils.data import DOMAINBED_DATASETS


def save_checkpoint(
    model, optimizer, epoch: int, accuracy: float, args, is_best: bool = False
):
    """Save model checkpoint."""
    suffix = "best" if is_best else "final"
    dataset_name = args.dataset.upper()

    # Include domain info in filename for domain-based datasets
    if dataset_name in DOMAINBED_DATASETS or dataset_name in ["PACS", "PACS_DEEPLAKE"]:
        if args.classic_split:
            # Classic ML setup - all domains used
            domains_str = (
                "_".join(sorted(args.train_domains)) if args.train_domains else "all"
            )
            filename = (
                f"{args.method}_{args.teacher}_to_{args.student}_{args.dataset}_"
                f"classic_{domains_str}_{suffix}.pth"
            )
        else:
            # OOD setup - separate train/val domains
            train_domains_str = (
                "_".join(sorted(args.train_domains)) if args.train_domains else "all"
            )
            val_domains_str = (
                "_".join(sorted(args.val_domains)) if args.val_domains else "default"
            )
            filename = (
                f"{args.method}_{args.teacher}_to_{args.student}_{args.dataset}_"
                f"train_{train_domains_str}_val_{val_domains_str}_{suffix}.pth"
            )
    else:
        filename = f"{args.method}_{args.teacher}_to_{args.student}_{args.dataset}_{suffix}.pth"

    save_path = Path(args.save_dir) / filename

    checkpoint = model.state_dict()
    torch.save(checkpoint, save_path)

    if is_best:
        print(f"New best accuracy: {accuracy:.2f}% - Saved to {save_path}")
