import itertools
import torch
from models.utils import sigma_vae_loss, masked_cross_entropy, kl_divergence
from models.mmvae import MMVAE

MODALITIES = ["image", "text", "action"]
# every non-empty subset of modalities, as in the reference repo
SUBSETS = [list(c) for r in range(1, len(MODALITIES) + 1) for c in itertools.combinations(MODALITIES, r)]

def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}

def make_subset(batch, keep):
    sub = dict(batch)
    for m in MODALITIES:
        if m not in keep:
            sub[m] = None
    return sub

def loss_all(out, batch, beta=1.0):
    l_img = sigma_vae_loss(out["image"],  batch["image_raw"])
    l_act = sigma_vae_loss(out["action"], batch["action_raw"], batch["action_mask"])
    l_txt = masked_cross_entropy(out["text"], batch["text"], batch["text_mask"]) # modification from paper
    kld = kl_divergence(out["mu"], out["logvar"])
    total = l_img + l_act + l_txt + beta * kld
    return total, {"img": l_img.item(), "act": l_act.item(),"txt": l_txt.item(), "kld": kld.item()}

def run_epoch(model, loader, optimizer, device, train=True, beta=1.0, amp=False):
    model.train() if train else model.eval()
    running = {"total": 0.0, "img": 0.0, "act": 0.0, "txt": 0.0, "kld": 0.0}
    n = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = to_device(batch, device)
            b = batch["batch_size"]

            if train:
                optimizer.zero_grad()

            total = 0.0
            for keep in SUBSETS:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    out = model(make_subset(batch, keep))
                    loss, parts = loss_all(out, batch, beta)
                if train:
                    loss.backward()  # per-subset backward keeps memory flat
                total += loss.item()
                if set(keep) == set(MODALITIES):
                    for k, v in parts.items():
                        running[k] += v * b

            if train:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running["total"] += total * b
            n += b

    return {k: v / n for k, v in running.items()}

def run(train_ld, val_ld, device, epochs=400, latent_dim=32, vocab_size=16, action_dim=4, lr=1e-3, amp=False, save_path="best.pt", beta=1.0):
    model = MMVAE(latent_dim, vocab_size, action_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = float("inf")
    for epoch in range(epochs):
        tr = run_epoch(model, train_ld, opt, device, train=True, amp=amp, beta=beta)
        va = run_epoch(model, val_ld, opt, device, train=False, amp=amp, beta=beta)

        print(f"{epoch:03d} | train {tr['total']:.4f} "
              f"(img {tr['img']:.3f} act {tr['act']:.3f} "
              f"txt {tr['txt']:.3f} kld {tr['kld']:.3f}) "
              f"| val {va['total']:.4f}")

        if va["total"] < best:
            best = va["total"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "val": best}, save_path)

    return model