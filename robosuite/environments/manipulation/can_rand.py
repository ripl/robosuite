from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CanObject, CanVisualObject, BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string


class CanRand(ManipulationEnv):
    """
    Single-table pick-and-place variant with one can and a visual can target.

    - One `CanObject` is placed randomly on the table at reset.
    - One `CanVisualObject` (semi-transparent can) indicates the target location, also placed randomly.
    - The target is sampled far enough from the can to avoid immediate success (controlled by
      `min_goal_separation`).

    Args:
        robots (str or list of str): Robot arm(s) to instantiate. Must be a single single-arm robot.
        env_configuration (str): Robot placement configuration (default: "default").
        controller_configs (dict or list of dict): Controller parameters for custom controllers. If None, uses
            task defaults.
        gripper_types (str or list of str): Gripper model(s) to attach. "default" uses robot defaults.
        base_types (str or list of str): Base model(s) to attach. "default" uses robot defaults.
        initialization_noise (dict or list of dict): Noise settings for initial joint positions. "default" uses
            task defaults.
        table_full_size (3-tuple): Table (x, y, z) full dimensions.
        table_friction (3-tuple): MuJoCo friction parameters (sliding, torsional, rolling) for table.
        use_camera_obs (bool): If True, include rendered image(s) in observations.
        use_object_obs (bool): If True, include object state in observations.
        reward_scale (float or None): Scales the task reward by this value.
        reward_shaping (bool): If True, adds optional dense shaping via `staged_rewards()`.
        has_renderer (bool): If True, renders to an on-screen viewer.
        has_offscreen_renderer (bool): If True, enables off-screen rendering.
        render_camera (str): Name of the camera used for rendering.
        render_collision_mesh (bool): If True, render collision meshes.
        render_visual_mesh (bool): If True, render visual meshes.
        render_gpu_device_id (int): GPU device id for offscreen rendering. -1 lets robosuite infer from env vars.
        control_freq (float): Control frequency in Hz.
        lite_physics (bool): If True, enables lite physics optimizations.
        horizon (int): Episode length in timesteps.
        ignore_done (bool): If True, never terminates episodes (ignores horizon).
        hard_reset (bool): If True, fully reloads model / sim on reset; else only resets state.
        camera_names (str or list of str): Camera name(s) to render.
        camera_heights (int or list of int): Height(s) of rendered frames.
        camera_widths (int or list of int): Width(s) of rendered frames.
        camera_depths (bool or list of bool): If True, render depth along with RGB.
        camera_segmentations (None or str or list): Segmentation type(s) (None, 'instance', 'class', 'element').
        renderer (str): Renderer backend.
        renderer_config (dict or None): Renderer-specific options.
        min_goal_separation (float): Minimum xy-separation between can and visual target at reset.
        place_tolerance (float): XY distance threshold for success when placing near the visual target.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
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
        min_goal_separation=0.15,
        place_tolerance=0.05,
    ):
        # table settings
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))

        # reward + obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        # placement + success parameters
        self.min_goal_separation = float(min_goal_separation)
        self.place_tolerance = float(place_tolerance)

        # object placement samplers are defined in _load_model
        self.can_sampler = None
        self.target_sampler = None

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
        """
        Sparse reward:

        - Returns 1.0 if the can is placed near the visual target (see `_check_success()`), else 0.0.
        - Scaled by `reward_scale` if provided.

        Args:
            action (np.array): Unused.

        Returns:
            float: Reward value.
        """
        r = 1.0 if self._check_success() else 0.0
        if self.reward_scale is not None:
            r *= self.reward_scale
        return r

    def staged_rewards(self):
        """
        Optional dense shaping terms, similar in spirit to other tasks. Useful when `reward_shaping=True`.

        Returns:
            tuple: (reach, grasp, lift, hover)
        """
        reach_mult = 0.1
        grasp_mult = 0.35
        hover_mult = 0.7

        # reaching
        dist = self._gripper_to_target(
            gripper=self.robots[0].gripper, target=self.can.root_body, target_type="body", return_distance=True
        )
        r_reach = (1 - np.tanh(10.0 * dist)) * reach_mult

        # grasping
        r_grasp = int(self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.can.contact_geoms)) * grasp_mult

        # hovering near target (xy distance small)
        can_xy = self.sim.data.body_xpos[self.can_body_id][:2]
        goal_xy = self.sim.data.body_xpos[self.target_body_id][:2]
        xy_d = np.linalg.norm(can_xy - goal_xy)
        r_hover = (1 - np.tanh(10.0 * max(xy_d - self.place_tolerance, 0.0))) * (hover_mult - grasp_mult)
        if r_grasp > 0:
            r_hover += grasp_mult

        return r_reach, r_grasp, 0.0, r_hover

    def _load_model(self):
        """Loads the MJCF model and constructs arena, objects, and task graph."""
        super()._load_model()

        # Adjust base pose for single table
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Make the robot mount / base invisible (hide all base geoms)
        base_model = self.robots[0].robot_model.base
        for geom in base_model.worldbody.iter("geom"):
            geom.set("rgba", array_to_string([0, 0, 0, 0]))

        # Table arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Make floor, walls, and table invisible
        mujoco_arena.floor.set("rgba", array_to_string([0, 0, 0, 0]))
        for geom in mujoco_arena.worldbody.iter("geom"):
            name = geom.get("name", "")
            if name.startswith("wall_"):
                geom.set("rgba", array_to_string([0, 0, 0, 0]))
        for geom in [mujoco_arena.table_visual, mujoco_arena.table_collision] + mujoco_arena.table_legs_visual:
            if geom is not None:
                geom.set("rgba", array_to_string([0, 0, 0, 0]))

        # Table-top white plane (visual-only), sized to table top
        plane_half_size = [
            float(mujoco_arena.table_half_size[0]),
            float(mujoco_arena.table_half_size[1]),
            0.001,
        ]
        self.plane = BoxObject(
            name="plane",
            size=plane_half_size,
            rgba=[1, 1, 1, 1],
            material=None,
            joints="default",
            density=500.0,
            obj_type="all",
            duplicate_collision_geoms=False,
        )

        self.plane._obj.set("pos", array_to_string([0, 0, float(self.table_offset[2])]))
        for g in self.plane._obj.iter("geom"):
            g.set("contype", "0")
            g.set("conaffinity", "0")
        self.plane._obj.set("gravcomp", "1")

        self.plane_sampler = UniformRandomSampler(
            name="PlaneSampler",
            mujoco_objects=self.plane,
            x_range=[-0.2, 0.2],
            y_range=[-0.2, 0.2],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=self.table_offset,
            z_offset=0.0,
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
            x_range=[-2.0, 2.0],
            y_range=[-2.0, 2.0],
            rotation=[0.0, 2 * np.pi],
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=np.array([0.0, 0.0, 0.0]),
            z_offset=0.0,
        )

        # Objects: one can + one visual target can (target is a copy with physics disabled)
        self.can = CanObject(name="Can")
        self.target = CanObject(name="VisualCan")
        # Make target visual-only and semi-transparent: disable contacts and set rgba
        for g in self.target._obj.iter("geom"):
            g.set("contype", "0")
            g.set("conaffinity", "0")
            # Remove material so RGBA is respected (prevents textured appearance)
            if "material" in g.attrib:
                del g.attrib["material"]
            g.set("rgba", array_to_string([0.8, 0.8, 0.8, 0.3]))
        # Keep mass / inertia from CanObject so free joint is valid, but counteract gravity
        self.target._obj.set("gravcomp", "1")

        # Placement ranges across the table with a margin
        self.can_sampler = UniformRandomSampler(
            name="CanSampler",
            mujoco_objects=self.can,
            x_range=[-0.20, 0.20],
            y_range=[-0.20, 0.20],
            rotation=None,
            rotation_axis="z",
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.target_sampler = UniformRandomSampler(
            name="TargetSampler",
            mujoco_objects=self.target,
            x_range=[-0.20, 0.20],
            y_range=[-0.20, 0.20],
            rotation=0.0,
            rotation_axis="z",
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.0,
        )

        # Task graph
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.target, self.can, self.plane, self.floor_plane],
        )

    def _setup_references(self):
        """Sets up mujoco IDs for the can and visual target bodies."""
        super()._setup_references()
        self.can_body_id = self.sim.model.body_name2id(self.can.root_body)
        self.target_body_id = self.sim.model.body_name2id(self.target.root_body)

    def _setup_observables(self):
        """Creates object observables (can pose) and gripper-relative sensors."""
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def can_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.can_body_id])

            @sensor(modality=modality)
            def can_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.can_body_id], to="xyzw")

            sensors = [can_pos, can_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors += [
                self._get_obj_eef_sensor(full_pf, "can_pos", f"{arm_pf}gripper_to_can_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables

    def _reset_internal(self):
        """Samples and applies placements for can, target, and planes on reset."""
        super()._reset_internal()

        if not self.deterministic_reset:
            # Sample can placement
            placements = self.can_sampler.sample()
            can_pos, can_quat, _ = placements[self.can.name]

            # Sample target placement with separation constraint
            goal_pos = None
            goal_quat = None
            for _ in range(5000):
                goal_sample = self.target_sampler.sample()
                g_pos, g_quat, _ = goal_sample[self.target.name]
                if np.linalg.norm(np.array(g_pos)[:2] - np.array(can_pos)[:2]) >= self.min_goal_separation:
                    goal_pos, goal_quat = g_pos, g_quat
                    break
            if goal_pos is None:
                # Fall back to the last sample (extremely unlikely to happen given ranges)
                goal_pos, goal_quat = g_pos, g_quat

            # Apply placements
            self.sim.data.set_joint_qpos(
                self.can.joints[0], np.concatenate([np.array(can_pos), np.array(can_quat)])
            )
            # Sample and apply placements for table plane and floor plane
            plane_placements = self.plane_sampler.sample()
            floor_plane_placements = self.floor_plane_sampler.sample()
            p_pos, p_quat, _ = plane_placements[self.plane.name]
            fp_pos, fp_quat, _ = floor_plane_placements[self.floor_plane.name]
            self.sim.data.set_joint_qpos(
                self.plane.joints[0], np.concatenate([np.array(p_pos), np.array(p_quat)])
            )
            self.sim.data.set_joint_qpos(
                self.floor_plane.joints[0], np.concatenate([np.array(fp_pos), np.array(fp_quat)])
            )
            # Place the visual target via its free joint so it's included in flattened state
            # Match target height to the actual can's height for clear visual alignment
            goal_pos = np.array(goal_pos)
            goal_pos[2] = float(can_pos[2])
            self.sim.data.set_joint_qpos(
                self.target.joints[0], np.concatenate([np.array(goal_pos), np.array(goal_quat)])
            )

    def _check_success(self):
        """Returns True if can is near visual target in XY and within table-height band."""
        can_p = self.sim.data.body_xpos[self.can_body_id]
        goal_p = self.sim.data.body_xpos[self.target_body_id]
        xy_close = np.linalg.norm(can_p[:2] - goal_p[:2]) <= self.place_tolerance

        # near table height band
        table_h = self.table_offset[2]
        z_ok = table_h < can_p[2] < table_h + 0.1

        return bool(xy_close and z_ok)

    def visualize(self, vis_settings):
        """Visualize gripper distance to the can in the viewer."""
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.can.root_body, target_type="body")


        
