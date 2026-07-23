
# Run:python data/collect_sort.py --episodes 2000 --out data/sort_demos.pkl

import argparse
import os
import pickle
import sys
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs 

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data.sort_env 

PAD = "<pad>"
DELTA_SCALE = 0.1
CUBES = ["red", "blue"]
BOXES = ["green", "yellow"]


def to_np(x):
    return x.squeeze(0).detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def scripted_sort(env, cube_color, box_color, max_steps=140):
    obs, _ = env.reset()
    base = env.unwrapped
    image = to_np(obs["sensor_data"]["base_camera"]["rgb"]).astype(np.uint8) # camera stuff

    cube = base.cube if cube_color == "red" else base.cube_blue
    other = base.cube_blue if cube_color == "red" else base.cube
    box = base.box_green if box_color == "green" else base.box_yellow

    cube0 = to_np(cube.pose.p).copy()
    other0 = to_np(other.pose.p).copy()
    box_xy = to_np(box.pose.p)[:2].copy()

    traj, grip, phase, hold = [], 1.0, "hover", 0

    for _ in range(max_steps):
        tcp = to_np(base.agent.tcp.pose.p)
        c = to_np(cube.pose.p)

        if phase == "hover":
            target = np.array([c[0], c[1], 0.12])
            if np.linalg.norm(tcp[:2] - c[:2]) < 0.008 and tcp[2] < 0.13:
                phase = "descend"
        if phase == "descend":
            target = np.array([c[0], c[1], c[2] + 0.002])
            if np.linalg.norm(tcp - target) < 0.008:
                phase = "grasp"
        if phase == "grasp":
            target = tcp
            grip = -1.0
            hold += 1
            if hold >= 6:
                phase = "lift"
        if phase == "lift":
            target = np.array([tcp[0], tcp[1], 0.15])
            if tcp[2] > 0.14:
                phase = "carry"
        if phase == "carry":
            target = np.array([box_xy[0], box_xy[1], 0.15])
            if np.linalg.norm(tcp[:2] - box_xy) < 0.01:
                phase = "lower"
        if phase == "lower":
            target = np.array([box_xy[0], box_xy[1], 0.08])
            if tcp[2] < 0.09:
                phase = "release"
        if phase == "release":
            target = tcp
            grip = 1.0
            hold += 1
            if hold >= 12:
                break

        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip((target - tcp) / DELTA_SCALE, -1.0, 1.0)
        action[3] = grip
        obs, *_ = env.step(action)
        tcp_now = to_np(base.agent.tcp.pose.p)
        traj.append([*tcp_now.tolist(), 1.0 if grip > 0 else 0.0])

    cube_f = to_np(cube.pose.p)
    other_f = to_np(other.pose.p)
    ok = (np.linalg.norm(cube_f[:2] - box_xy) < 0.05  
          and cube_f[2] < 0.05         
          and np.linalg.norm(other_f[:2] - other0[:2]) < 0.03)
    return image, traj, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--out", type=str, default="data/sort_demos.pkl")
    args = ap.parse_args()

    env = gym.make("SortCubes-v1", obs_mode="rgb", control_mode="pd_ee_delta_pos")
    rng = np.random.default_rng(0)
    samples, tried = [], 0
    per_class = {}

    while len(samples) < args.episodes:
        tried += 1
        cube_color = CUBES[rng.integers(2)]
        box_color = BOXES[rng.integers(2)]
        image, traj, ok = scripted_sort(env, cube_color, box_color)
        if not ok or len(traj) < 10:
            continue
        words = f"put the {cube_color} cube in the {box_color} box".split()

        # getting the samples
        samples.append({"image": image, "words": words, "traj": traj, "task": f"{cube_color}->{box_color}"})
        k = f"{cube_color}->{box_color}"
        per_class[k] = per_class.get(k, 0) + 1
        if len(samples) % 100 == 0:
            print(f"{len(samples)}/{args.episodes} kept "f"({tried} tried, {len(samples)/tried:.0%} success) {per_class}")

    # the vocabulary
    vocab_words = sorted({w for s in samples for w in s["words"]})
    vocab = {w: i for i, w in enumerate([PAD] + vocab_words)}
    for s in samples:
        s["tokens"] = [vocab[w] for w in s.pop("words")]

    with open(args.out, "wb") as f:
        pickle.dump({"samples": samples, "vocab": vocab}, f)
    lens = [len(s["traj"]) for s in samples]
    print(f"saved {len(samples)} demos to {args.out} | vocab {len(vocab)} | "
          f"len min/med/max {min(lens)}/{int(np.median(lens))}/{max(lens)} | {per_class}")
    env.close()


if __name__ == "__main__":
    main()
