import argparse
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import data.sort_env  # noqa: F401  registers SortCubes-v1
from data.dataset import MVAEDataset, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from models.mmvae import MMVAE

DELTA_SCALE = 0.1  # meters per unit action for pd_ee_delta_pos


def to_np(x):
    return x.squeeze(0).detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def encode_image(rgb, device, size=IMG_SIZE):
    img = torch.tensor(rgb.copy()).permute(2, 0, 1).float() / 255.0
    if img.shape[-1] != size:
        img = F.interpolate(img.unsqueeze(0), size=(size, size),mode="bilinear", align_corners=False).squeeze(0)
    return ((img - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="best.pt")
    ap.add_argument("--data", type=str, default="data/sort_demos_5k.pkl",help="training pkl (for vocab + action normalization stats)")
    ap.add_argument("--env", type=str, default="PickCube-v1")
    ap.add_argument("--instruction", type=str, default="pick up the cube")
    ap.add_argument("--steps", type=int, default=80, help="length of the generated trajectory")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--cube-pos", type=float, nargs=2, default=None, metavar=("X", "Y"),help="place the cube at this xy (meters, table frame) after reset")
    ap.add_argument("--record", action="store_true",help="save an mp4 of each episode to videos/")
    ap.add_argument("--view", action="store_true", help="open the live SAPIEN viewer window (overrides --record)")
    ap.add_argument("--img-size", type=int, default=64, help="encoder input resolution (match training)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MVAEDataset(args.data)

    words = args.instruction.lower().split()
    unknown = [w for w in words if w not in ds.vocab]
    assert not unknown, f"words not in training vocab {sorted(ds.vocab)}: {unknown}"
    tokens = torch.tensor([[ds.vocab[w] for w in words]], device=device)
    text_mask = torch.ones_like(tokens, dtype=torch.bool)

    model = MMVAE(latent_dim=args.latent_dim, vocab_size=ds.vocab_size, action_dim=4)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch')}, val {ckpt.get('val'):.3f})")

    env = gym.make(args.env, obs_mode="rgb", control_mode="pd_ee_delta_pos",
                   render_mode="human" if args.view else "rgb_array")
    if args.record and not args.view:
        from mani_skill.utils.wrappers import RecordEpisode
        env = RecordEpisode(env, output_dir="videos", save_video=True,
                            save_trajectory=False, video_fps=20)
    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if args.cube_pos is not None:
            import sapien
            cube = getattr(env.unwrapped, "cube", None) or getattr(env.unwrapped, "obj")
            p = to_np(cube.pose.p)
            q = to_np(cube.pose.q)
            cube.set_pose(sapien.Pose([args.cube_pos[0], args.cube_pos[1], p[2]], q))
            obs = env.unwrapped.get_obs()  # re-render with the moved cube
        rgb = to_np(obs["sensor_data"]["base_camera"]["rgb"])

        with torch.no_grad():
            batch = {"batch_size": 1, "image": encode_image(rgb, device, args.img_size),
                     "text": tokens, "text_mask": text_mask,
                     "action": None, "action_mask": None}
            mu, logvar = model.infer(batch)  # posterior from image + language only
            act_mask = torch.ones(1, args.steps, dtype=torch.bool, device=device)
            traj = model.dec_act(mu, act_mask)[0] # use posterior mean
            traj = ds.denormalize_action(traj).cpu().numpy()  # (steps, 4) abs xyz+grip

        info = {}
        for wp in traj:
            tcp = to_np(env.unwrapped.agent.tcp.pose.p)
            action = np.zeros(4, dtype=np.float32)
            action[:3] = np.clip((wp[:3] - tcp) / DELTA_SCALE, -1.0, 1.0)
            action[3] = 1.0 if wp[3] > 0.5 else -1.0
            obs, _, _, _, info = env.step(action)
            if args.view:
                env.render()
        ok = bool(to_np(info.get("success", torch.tensor(False))))
        successes += ok
        base = env.unwrapped
        obj = getattr(base, "cube", None) or getattr(base, "obj", None)
        goal = getattr(base, "goal_region", None) or getattr(base, "goal_site", None)
        dist = ""
        if obj is not None and goal is not None:
            d = np.linalg.norm(to_np(obj.pose.p)[:2] - to_np(goal.pose.p)[:2])
            dist = f"cube->goal center: {d*100:.1f} cm"
        print(f"episode {ep}: success={ok}{dist}")

    print(f"{successes}/{args.episodes} successful")
    env.close()


if __name__ == "__main__":
    main()
