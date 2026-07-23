# Batch dict [image / image_raw / text / text_mask / action / action_raw / action_mask / batch_size.]
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMG_SIZE = 64


class MVAEDataset(Dataset):
    def __init__(self, path, img_size=IMG_SIZE):
        self.img_size = img_size
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.samples = d["samples"]
        self.vocab = d["vocab"]
        self.vocab_size = len(self.vocab)
        self.max_txt = max(len(s["tokens"]) for s in self.samples)
        self.max_act = max(len(s["traj"]) for s in self.samples)

        # normalize xyz per-dim (gripper stays binary)
        steps = torch.cat([torch.tensor(s["traj"], dtype=torch.float32) for s in self.samples])
        self.act_mean = steps.mean(0)
        self.act_std = steps.std(0).clamp(min=1e-6)
        self.act_mean[3], self.act_std[3] = 0.0, 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]

        raw = torch.tensor(s["image"]).permute(2, 0, 1).float() / 255.0

        def resize(t, size):
            if t.shape[-1] == size:
                return t
            return F.interpolate(t.unsqueeze(0), size=(size, size),
                                 mode="bilinear", align_corners=False).squeeze(0)

        img = resize(raw, self.img_size) # encoder input
        img64 = resize(raw, IMG_SIZE) # decoder recon target (64x64)

        txt = torch.zeros(self.max_txt, dtype=torch.long)
        txt[:len(s["tokens"])] = torch.tensor(s["tokens"])
        txt_mask = torch.zeros(self.max_txt, dtype=torch.bool)
        txt_mask[:len(s["tokens"])] = True

        traj = (torch.tensor(s["traj"], dtype=torch.float32) - self.act_mean) / self.act_std
        act = torch.zeros(self.max_act, 4)
        act[:len(traj)] = traj
        act_mask = torch.zeros(self.max_act, dtype=torch.bool)
        act_mask[:len(traj)] = True

        return {
            "image": (img - IMAGENET_MEAN) / IMAGENET_STD,  # encoder input
            "image_raw": img64, 
            "text": txt, "text_mask": txt_mask,
            "action": act, "action_raw": act.clone(),
            "action_mask": act_mask,
        }

    def denormalize_action(self, traj):
        # (L, 4) normalized -> absolute [x, y, z, gripper] for execution.
        return traj * self.act_std.to(traj.device) + self.act_mean.to(traj.device)


def collate(items):
    batch = {k: torch.stack([it[k] for it in items]) for k in items[0]}
    batch["batch_size"] = len(items)
    return batch
