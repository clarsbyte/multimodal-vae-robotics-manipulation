import argparse
import torch
from torch.utils.data import DataLoader, random_split
from data.dataset import MVAEDataset, collate
from models.train import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/maniskill_demos.pkl")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-split", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true", help="bf16 mixed precision (CUDA)")
    ap.add_argument("--save", type=str, default="best.pt", help="checkpoint path")
    ap.add_argument("--img-size", type=int, default=64, help="encoder input resolution")
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MVAEDataset(args.data, img_size=args.img_size)
    n_val = max(1, int(len(ds) * args.test_split))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,collate_fn=collate, num_workers=2)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    print(f"{len(train_ds)} train / {len(val_ds)} val | vocab {ds.vocab_size} "f"| max traj len {ds.max_act} | device {device}")
    run(train_ld, val_ld, device, epochs=args.epochs, latent_dim=args.latent_dim,
        vocab_size=ds.vocab_size, action_dim=4, lr=args.lr, amp=args.amp,
        save_path=args.save, beta=args.beta)


if __name__ == "__main__":
    main()
