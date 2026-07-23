import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
import mani_skill.envs.utils.randomization as randomization

PAD_HALF = 0.05       
MIN_CUBE_CUBE = 0.07
MIN_CUBE_PAD = 0.10
CUBE_X, CUBE_Y = 0.10, 0.15 
PAD_X = 0.08


@register_env("SortCubes-v1", max_episode_steps=150)
class SortCubesEnv(PickCubeEnv):
    def _load_scene(self, options: dict):
        super()._load_scene(options) # table, red self.cube, hidden goal site
        self.cube_blue = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[0, 0, 1, 1],
            name="cube_blue", initial_pose=sapien.Pose(p=[0.1, 0.1, self.cube_half_size]))
        self.box_green = actors.build_box(
            self.scene, half_sizes=[PAD_HALF, PAD_HALF, 0.005], color=[0, 0.7, 0, 1],
            name="box_green", body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, -0.12, 0.005]))
        self.box_yellow = actors.build_box(
            self.scene, half_sizes=[PAD_HALF, PAD_HALF, 0.005], color=[0.9, 0.8, 0, 1],
            name="box_yellow", body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0.12, 0.005]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            rng = np.random.default_rng()

            def sample_all():
                # pads in opposite y-halves, cubes anywhere clear of pads
                green = np.array([rng.uniform(-PAD_X, PAD_X), rng.uniform(-0.18, -0.07)])
                yellow = np.array([rng.uniform(-PAD_X, PAD_X), rng.uniform(0.07, 0.18)])
                cubes = []
                while len(cubes) < 2:
                    c = np.array([rng.uniform(-CUBE_X, CUBE_X), rng.uniform(-CUBE_Y, CUBE_Y)])
                    if (np.linalg.norm(c - green) > MIN_CUBE_PAD and np.linalg.norm(c - yellow) > MIN_CUBE_PAD and all(np.linalg.norm(c - o) > MIN_CUBE_CUBE for o in cubes)):
                        cubes.append(c)
                return green, yellow, cubes

            for i in range(b):
                green, yellow, (red_xy, blue_xy) = sample_all()
                z = self.cube_half_size
                q = randomization.random_quaternions(1, lock_x=True, lock_y=True)
                self.cube.set_pose(Pose.create_from_pq(
                    torch.tensor([[red_xy[0], red_xy[1], z]], dtype=torch.float32), q))
                q = randomization.random_quaternions(1, lock_x=True, lock_y=True)
                self.cube_blue.set_pose(Pose.create_from_pq(
                    torch.tensor([[blue_xy[0], blue_xy[1], z]], dtype=torch.float32), q))
                self.box_green.set_pose(Pose.create_from_pq(
                    torch.tensor([[green[0], green[1], 0.005]], dtype=torch.float32)))
                self.box_yellow.set_pose(Pose.create_from_pq(
                    torch.tensor([[yellow[0], yellow[1], 0.005]], dtype=torch.float32)))
