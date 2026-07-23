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
from eval_sort import calibrate_pix2world, assist_traj, DELTA_SCALE
from models.mmvae import MMVAE  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="best_sort.pt")
    ap.add_argument("--data", type=str, default="data/sort_demos_5k.pkl")
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--no-view", action="store_true", help="run headless")
    ap.add_argument("--no-assist", action="store_true", help="pure end-to-end model")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MVAEDataset(args.data, img_size=args.img_size)
    model = MMVAE(latent_dim=args.latent_dim, vocab_size=ds.vocab_size, action_dim=4)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    env = gym.make("SortCubes-v1", obs_mode="rgb", control_mode="pd_ee_delta_pos", render_mode="rgb_array" if args.no_view else "human")
    H = None if args.no_assist else calibrate_pix2world(env)

    vocab_words = sorted(w for w in ds.vocab if w != "<pad>")
    print(f"\nready. vocab: {' '.join(vocab_words)}")
    print('example: "put the red cube in the yellow box" | "new" = new scene | "quit"\n')

    obs, _ = env.reset()
    scene_seed = 0
    while True:
        if not args.no_view:
            env.render()
        try:
            line = input("> ").strip().lower()
        except EOFError:
            break
        if line in ("quit", "exit", "q"):
            break
        if line in ("new", "reset", "n"):
            scene_seed += 1
            obs, _ = env.reset(seed=scene_seed)
            print("new scene.")
            continue
        words = line.split()
        unknown = [w for w in words if w not in ds.vocab]
        if unknown:
            print(f"unknown words {unknown} | usable words: {' '.join(vocab_words)}")
            continue
        cube_color = next((w for w in words if w in ("red", "blue")), None)
        box_color = next((w for w in words if w in ("green", "yellow")), None)

        base = env.unwrapped
        rgb = to_np(obs["sensor_data"]["base_camera"]["rgb"])
        toks = torch.tensor([[ds.vocab[w] for w in words]], device=device)

        #execute inference
        with torch.no_grad():
            mu, _ = model.infer({"batch_size": 1,
                                 "image": encode_image(rgb, device, args.img_size),
                                 "text": toks,
                                 "text_mask": torch.ones_like(toks, dtype=torch.bool),
                                 "action": None})
            mask = torch.ones(1, args.steps, dtype=torch.bool, device=device)
            traj = ds.denormalize_action(model.dec_act(mu, mask)[0]).cpu().numpy()
        if H is not None:
            traj = assist_traj(traj, rgb, H, cube_color=cube_color, box_color=box_color)

        for wp in traj:
            tcp = to_np(base.agent.tcp.pose.p)
            a = np.zeros(4, dtype=np.float32)
            a[:3] = np.clip((wp[:3] - tcp) / DELTA_SCALE, -1.0, 1.0)
            a[3] = 1.0 if wp[3] > 0.5 else -1.0
            obs, *_ = env.step(a)
            if not args.no_view:
                env.render()

        if cube_color and box_color:
            cube = base.cube if cube_color == "red" else base.cube_blue
            box = base.box_green if box_color == "green" else base.box_yellow
            d = np.linalg.norm(to_np(cube.pose.p)[:2] - to_np(box.pose.p)[:2])
            print(f"  {cube_color} cube -> {box_color} box: {d*100:.1f} cm from center "
                  f"({'IN THE BOX' if d < 0.06 else 'missed'})")
        else:
            print("executed (no cube/box colors found in instruction to score against)")

    env.close()


if __name__ == "__main__":
    main()
