import argparse
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data.sort_env  
from data.dataset import MVAEDataset
from infer import encode_image, to_np
from models.mmvae import MMVAE

DELTA_SCALE = 0.1


def largest_blob(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def detect_centers(rgb):
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    masks = {
        "red":    (r > 150) & (g < 60) & (b < 60),
        "blue":   (b > 110) & (r < 70) & (g < 110),
        "green":  (g > 90) & (r < 70) & (b < 70),
        "yellow": (r > 185) & (g > 160) & (b < 60),
    }
    out = {}
    for k, m in masks.items():
        blob = largest_blob(m)
        if blob is not None and blob.sum() >= 4:
            ys, xs = np.nonzero(blob)
            out[k] = np.array([xs.mean(), ys.mean()])
    return out


def fit_homography(P, W):
    P, W = np.array(P), np.array(W)
    M = []
    for (px, py), (wx, wy) in zip(P, W):
        M.append([px, py, 1, 0, 0, 0, -wx * px, -wx * py, -wx])
        M.append([0, 0, 0, px, py, 1, -wy * px, -wy * py, -wy])
    _, _, Vt = np.linalg.svd(np.array(M))
    return Vt[-1].reshape(3, 3)


def apply_h(H, p):
    v = H @ np.array([p[0], p[1], 1.0])
    return v[:2] / v[2]

#table x-y homographies
def calibrate_pix2world(env, n=20):
    pts = {"cube": ([], []), "pad": ([], [])}
    for i in range(n):
        obs, _ = env.reset(seed=90000 + i)
        rgb = to_np(obs["sensor_data"]["base_camera"]["rgb"])
        det = detect_centers(rgb)
        base = env.unwrapped
        for color, actor, plane in [("red", base.cube, "cube"), ("blue", base.cube_blue, "cube"),("green", base.box_green, "pad"), ("yellow", base.box_yellow, "pad")]:
            if color in det:
                pts[plane][0].append(det[color])
                pts[plane][1].append(to_np(actor.pose.p)[:2])
    H = {plane: fit_homography(P, W) for plane, (P, W) in pts.items()}
    for plane, (P, W) in pts.items():
        err = np.array([np.linalg.norm(apply_h(H[plane], p) - w) for p, w in zip(P, W)])
        print(f"calibration [{plane}]: n={len(P)} mean err {err.mean()*100:.2f} cm")
    return H


def assist_traj(traj, rgb, H, cube_color=None, box_color=None):
    det_px = detect_centers(rgb)
    det = {k: apply_h(H["cube" if k in ("red", "blue") else "pad"], v)
           for k, v in det_px.items()}
    closed = traj[:, 3] < 0.5
    if not closed.any():
        return traj
    close_idx = int(np.argmax(closed))
    open_after = np.nonzero(~closed[close_idx:])[0]
    release_idx = close_idx + int(open_after[0]) if len(open_after) else len(traj) - 1

    grasp_xy = traj[max(close_idx - 1, 0), :2]
    place_xy = traj[max(release_idx - 1, 0), :2]

    if cube_color is not None:
        cube_t = det.get(cube_color)
    else:
        cubes = [det[k] for k in ("red", "blue") if k in det]
        cube_t = min(cubes, key=lambda c: np.linalg.norm(grasp_xy - c)) if cubes else None
    if box_color is not None:
        box_t = det.get(box_color)
    else:
        boxes = [det[k] for k in ("green", "yellow") if k in det]
        box_t = min(boxes, key=lambda c: np.linalg.norm(place_xy - c)) if boxes else None
    if cube_t is None or box_t is None:
        return traj

    out = traj.copy()
    out[:close_idx + 2, :2] += cube_t - grasp_xy
    out[close_idx + 2:, :2] += box_t - place_xy
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="best_sort.pt")
    ap.add_argument("--data", type=str, default="data/sort_demos_5k.pkl")
    ap.add_argument("--episodes", type=int, default=10, help="scenes; x4 instructions each")
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--assist", action="store_true",
                    help="snap grasp/place waypoints to color-detected objects "
                         "(nearest to the model's own waypoints)")
    ap.add_argument("--assist-instr", action="store_true",
                    help="like --assist, but the instruction's color words pick "
                         "the snap targets")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MVAEDataset(args.data)
    model = MMVAE(latent_dim=args.latent_dim, vocab_size=ds.vocab_size, action_dim=4)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch')}, val {ckpt.get('val'):.1f})")

    env = gym.make("SortCubes-v1", obs_mode="rgb", control_mode="pd_ee_delta_pos")
    A = calibrate_pix2world(env) if (args.assist or args.assist_instr) else None
    combos = [(c, b) for c in ["red", "blue"] for b in ["green", "yellow"]]
    stats = {f"{c}->{b}": [0, 0] for c, b in combos}
    wrong_cube = 0

    for ep in range(args.episodes):
        for cube_color, box_color in combos:
            obs, _ = env.reset(seed=1000 + ep)  
            base = env.unwrapped
            cube = base.cube if cube_color == "red" else base.cube_blue
            other = base.cube_blue if cube_color == "red" else base.cube
            box = base.box_green if box_color == "green" else base.box_yellow
            other0 = to_np(other.pose.p).copy()
            box_xy = to_np(box.pose.p)[:2].copy()

            instr = f"put the {cube_color} cube in the {box_color} box"
            toks = torch.tensor([[ds.vocab[w] for w in instr.split()]], device=device)
            rgb = to_np(obs["sensor_data"]["base_camera"]["rgb"])
            with torch.no_grad():
                batch = {"batch_size": 1, "image": encode_image(rgb, device, args.img_size),
                         "text": toks,
                         "text_mask": torch.ones_like(toks, dtype=torch.bool),
                         "action": None, "action_mask": None}
                mu, _ = model.infer(batch)
                mask = torch.ones(1, args.steps, dtype=torch.bool, device=device)
                traj = ds.denormalize_action(model.dec_act(mu, mask)[0]).cpu().numpy()
            if A is not None:
                traj = assist_traj(traj, rgb, A,
                                   cube_color=cube_color if args.assist_instr else None,
                                   box_color=box_color if args.assist_instr else None)

            for wp in traj:
                tcp = to_np(base.agent.tcp.pose.p)
                a = np.zeros(4, dtype=np.float32)
                a[:3] = np.clip((wp[:3] - tcp) / DELTA_SCALE, -1.0, 1.0)
                a[3] = 1.0 if wp[3] > 0.5 else -1.0
                env.step(a)

            cube_f = to_np(cube.pose.p)
            other_f = to_np(other.pose.p)
            ok = (np.linalg.norm(cube_f[:2] - box_xy) < 0.06 and cube_f[2] < 0.06
                  and np.linalg.norm(other_f[:2] - other0[:2]) < 0.04)
            if np.linalg.norm(other_f[:2] - other0[:2]) > 0.04:
                wrong_cube += 1
            k = f"{cube_color}->{box_color}"
            stats[k][0] += ok
            stats[k][1] += 1

    print()
    total_ok = total_n = 0
    for k, (ok, n) in stats.items():
        print(f"  {k:14s} {ok}/{n}")
        total_ok += ok
        total_n += n
    print(f"overall: {total_ok}/{total_n} ({total_ok/total_n:.0%})  "
          f"| wrong-cube-disturbed episodes: {wrong_cube}")
    env.close()


if __name__ == "__main__":
    main()
