from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import DoorObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import array_to_string, CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler


class DoorRand(ManipulationEnv):
    """
    Door task with randomized door XY placement on a 180-degree arc in front of the robot.

    - The door's XY is sampled on a circle arc centered at the robot base, spanning 180 degrees in front
      of the arm (theta in [-pi/2, pi/2] w.r.t. +X from the robot base).
    - The radial distance is within ±0.10 m of the default Door environment distance.
    - The door yaw is set so the door plane is tangent to the arc and the handle faces the robot.

    Args mirror `Door` unless noted otherwise.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        use_latch=True,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        radial_jitter=0.10,  # ±10 cm around default Door distance
    ):
        # Table settings (same as Door)
        self.table_full_size = (0.8, 0.3, 0.05)
        self.table_offset = (-0.2, -0.35, 0.8)

        # Reward + obs
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        # Arc sampling config
        self.radial_jitter = float(radial_jitter)
        self._base_xy = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        reward = 0.0
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            dist = np.linalg.norm(self._gripper_to_handle)
            reward += 0.25 * (1 - np.tanh(10.0 * dist))
            if self.use_latch:
                handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
                reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _load_model(self):
        super()._load_model()

        # Place robot base (same policy as Door)
        base_pos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(base_pos)
        self._base_xy = np.array(base_pos[:2], dtype=float)

        # Arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Make robot base invisible
        base_model = self.robots[0].robot_model.base
        for geom in base_model.worldbody.iter("geom"):
            geom.set("rgba", array_to_string([0, 0, 0, 0]))

        # Make floor, walls, and table invisible
        mujoco_arena.floor.set("rgba", array_to_string([0, 0, 0, 0]))
        for geom in mujoco_arena.worldbody.iter("geom"):
            name = geom.get("name", "")
            if name.startswith("wall_"):
                geom.set("rgba", array_to_string([0, 0, 0, 0]))
        for geom in [mujoco_arena.table_visual, mujoco_arena.table_collision] + mujoco_arena.table_legs_visual:
            if geom is not None:
                geom.set("rgba", array_to_string([0, 0, 0, 0]))

        # Door object
        self.door = DoorObject(
            name="Door",
            friction=0.0,
            damping=0.1,
            lock=self.use_latch,
        )

        # Floor plane carrying floor texture (visual-only)
        import os, robosuite
        floor_tex_path = os.path.join(
            os.path.dirname(robosuite.__file__),
            "models",
            "assets",
            "textures",
            "light-gray-floor-tile.png",
        )
        floor_plane_material = CustomMaterial(
            texture=floor_tex_path,
            tex_name="floor_tex",
            mat_name="floor_plane_mat",
            tex_attrib={"type": "2d"},
            mat_attrib={"texrepeat": "20 20", "specular": "0.0", "shininess": "0.0"},
        )

        self.floor_plane = BoxObject(
            name="floor_plane",
            size=[10, 10, 0.001],
            rgba=None,
            material=floor_plane_material,
            joints="default",
            density=500.0,
            obj_type="all",
            duplicate_collision_geoms=False,
        )

        self.floor_plane._obj.set("pos", array_to_string([0, 0, 0.0]))
        for g in self.floor_plane._obj.iter("geom"):
            g.set("contype", "0")
            g.set("conaffinity", "0")
        self.floor_plane._obj.set("gravcomp", "1")

        self.floor_plane_sampler = UniformRandomSampler(
            name="FloorPlaneSampler",
            mujoco_objects=self.floor_plane,
            x_range=[-0.05, 0.05],
            y_range=[-0.05, 0.05],
            rotation=[0.0, 2 * np.pi],
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=np.array([0.0, 0.0, 0.0]),
            z_offset=0.0,
        )

        # Task graph
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.door, self.floor_plane],
        )

    def _setup_references(self):
        super()._setup_references()
        self.object_body_ids = dict()
        self.object_body_ids["door"] = self.sim.model.body_name2id(self.door.door_body)
        self.object_body_ids["frame"] = self.sim.model.body_name2id(self.door.frame_body)
        self.object_body_ids["latch"] = self.sim.model.body_name2id(self.door.latch_body)
        self.door_handle_site_id = self.sim.model.site_name2id(self.door.important_sites["handle"])
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[0])
        if self.use_latch:
            self.handle_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[1])

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def door_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_body_ids["door"]])

            @sensor(modality=modality)
            def handle_pos(obs_cache):
                return self._handle_xpos

            @sensor(modality=modality)
            def hinge_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors = [door_pos, handle_pos, hinge_qpos]
            sensors += [
                self._get_obj_eef_sensor(full_pf, "door_pos", f"door_to_{arm_pf}eef_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            sensors += [
                self._get_obj_eef_sensor(full_pf, "handle_pos", f"handle_to_{arm_pf}eef_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]
            if self.use_latch:
                @sensor(modality=modality)
                def handle_qpos(obs_cache):
                    return np.array([self.sim.data.qpos[self.handle_qpos_addr]])
                sensors.append(handle_qpos)
                names.append("handle_qpos")
            for name, s in zip(names, sensors):
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            # Compute default Door radial distance from base (approximate default Door placement)
            default_local_xy = np.array([0.08, 0.0])  # mid of [0.07, 0.09] along +x
            default_world_xy = np.array(self.table_offset[:2]) + default_local_xy
            r_default = float(np.linalg.norm(default_world_xy - self._base_xy))
            r_default = r_default * 1.5

            # Sample radius within ±radial_jitter
            r_low = max(0.05, r_default - self.radial_jitter)
            r_high = r_default + self.radial_jitter
            r = np.random.uniform(low=r_low, high=r_high)

            # Sample theta on 180-degree arc in front of robot (+X direction)
            theta = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0)

            # Position on arc
            door_xy = self._base_xy + np.array([r * np.cos(theta), r * np.sin(theta)])
            door_z = float(self.table_offset[2] - self.door.bottom_offset[-1])
            door_pos = np.array([door_xy[0], door_xy[1], door_z])

            # Flip yaw by 180° relative to previous orientation
            yaw = float(theta)
            qw = np.cos(0.5 * yaw)
            qz = np.sin(0.5 * yaw)
            door_quat = np.array([qw, 0.0, 0.0, qz])  # (w,x,y,z)

            # Apply pose directly to door root body
            door_body_id = self.sim.model.body_name2id(self.door.root_body)
            self.sim.model.body_pos[door_body_id] = door_pos
            self.sim.model.body_quat[door_body_id] = door_quat

            # Sample and apply floor plane placement
            floor_plane_placements = self.floor_plane_sampler.sample()
            fp_pos, fp_quat, _ = floor_plane_placements[self.floor_plane.name]
            self.sim.data.set_joint_qpos(
                self.floor_plane.joints[0], np.concatenate([np.array(fp_pos), np.array(fp_quat)])
            )

    def _check_success(self):
        hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        return bool(hinge_qpos > 0.3)

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.door.important_sites["handle"], target_type="site"
            )

    @property
    def _handle_xpos(self):
        return self.sim.data.site_xpos[self.door_handle_site_id]

    @property
    def _gripper_to_handle(self):
        dists = []
        for arm in self.robots[0].arms:
            diff = self._handle_xpos - np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])
            dists.append(np.linalg.norm(diff))
        return min(dists)


