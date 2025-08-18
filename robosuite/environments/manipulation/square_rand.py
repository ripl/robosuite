import numpy as np
import xml.etree.ElementTree as ET

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import SquareNutObject, BoxObject, CompositeBodyObject
from robosuite.models.objects.primitive.cylinder import CylinderObject
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string


class SquareRand(ManipulationEnv):
    """
    Single-table setup with only a square nut and a vertical peg.

    - Square position (x, y) randomized in [-0.20, 0.20], yaw randomized
    - Peg position (x, y) randomized in [-0.20, 0.20], vertical, no initial collision
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
        table_offset=(0, 0, 0.8),
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
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

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
        # Sparse reward: 1.0 if task succeeded, else 0.0
        r = 1.0 if self._check_success() else 0.0
        if self.reward_scale is not None:
            r *= self.reward_scale
        return r

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        # Make robot base invisible
        base_model = self.robots[0].robot_model.base
        for geom in base_model.worldbody.iter("geom"):
            geom.set("rgba", array_to_string([0, 0, 0, 0]))

        # Make floor and walls invisible
        arena.floor.set("rgba", array_to_string([0, 0, 0, 0]))
        for geom in arena.worldbody.iter("geom"):
            name = geom.get("name", "")
            if name.startswith("wall_"):
                geom.set("rgba", array_to_string([0, 0, 0, 0]))

        # Make table invisible (top, collision, and legs)
        table_geoms = [arena.table_visual, arena.table_collision] + getattr(arena, "table_legs_visual", [])
        for g in table_geoms:
            if g is not None:
                g.set("rgba", array_to_string([0, 0, 0, 0]))

        # Add table-top white plane (visual-only) aligned with the table
        plane_half_size = [
            float(arena.table_half_size[0]),
            float(arena.table_half_size[1]),
            0.001,
        ]
        self.table_plane = BoxObject(
            name="table_plane",
            size=plane_half_size,
            rgba=[1, 1, 1, 1],
            material=None,
            joints="default",
            density=500.0,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.table_plane._obj.set("pos", array_to_string([0, 0, float(self.table_offset[2])]))
        for g in self.table_plane._obj.iter("geom"):
            g.set("contype", "0")
            g.set("conaffinity", "0")
        self.table_plane._obj.set("gravcomp", "1")

        # Add floor plane with texture (visual-only)
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

        # Randomizers for planes (match CanRand behavior)
        self.table_plane_sampler = UniformRandomSampler(
            name="TablePlaneSampler",
            mujoco_objects=self.table_plane,
            x_range=[-0.2, 0.2],
            y_range=[-0.2, 0.2],
                        rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=self.table_offset,
            z_offset=0.0,
        )

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

        # Objects: square nut (free) and peg assembly (cylinder + base as one moving body)
        self.square = SquareNutObject(name="Square")
        peg_cyl = CylinderObject(
            name="PegCyl",
            size=[0.012, 0.05],  # radius, half-length
            joints="default",
            rgba=[0.6, 0.6, 0.6, 1.0],
            obj_type="all",
            density=2000.0,
        )
        peg_base = BoxObject(
            name="PegBase",
            size=[0.06, 0.06, 0.005],
            rgba=[0.5, 0.5, 0.5, 1.0],
            joints=None,
            obj_type="all",
            density=2000.0,
        )
        # Set collision masks on child parts before composing
        for g in peg_cyl._obj.iter("geom"):
            g.set("contype", "2")
            g.set("conaffinity", "1")
        for g in peg_base._obj.iter("geom"):
            g.set("contype", "2")
            g.set("conaffinity", "1")
        # Build composite peg with a free joint so it's in flattened state and moves as one
        self.peg = CompositeBodyObject(
            name="Peg",
            objects=[peg_cyl, peg_base],
            object_locations=[
                [0.0, 0.0, 0.055],   # cylinder center above base (0.05 half-length + 0.005 base half-thickness)
                [0.0, 0.0, 0.0],     # base centered at origin
            ],
            object_quats=None,
            object_parents=None,
            joints="default",
        )
        # Do NOT account for the base in XY separation during placement: use cylinder radius only
        self.peg._horizontal = peg_cyl.horizontal_radius

        # Ensure composite peg geoms collide with square but not robot arm
        for g in self.peg._obj.iter("geom"):
            g.set("contype", "2")
            g.set("conaffinity", "1")
        # - Square: allow collisions with peg bit by including bit 2 in conaffinity (1|2 = 3)
        for g in self.square._obj.iter("geom"):
            g.set("conaffinity", "3")

        # One sampler for both; ensures non-overlap and uniform xy in [-0.2, 0.2]
        self.sampler = UniformRandomSampler(
            name="SquarePegSampler",
            mujoco_objects=[self.square, self.peg],
            x_range=[-0.18, 0.18],
            y_range=[-0.18, 0.18],
            rotation=None,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
            z_offset=0.0,
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.square, self.peg, self.table_plane, self.floor_plane],
        )

    def _setup_references(self):
        super()._setup_references()
        self.square_body_id = self.sim.model.body_name2id(self.square.root_body)
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)
        # No mocap anchor; peg is simply very heavy with a free joint

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def square_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.square_body_id])

            @sensor(modality=modality)
            def square_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.square_body_id], to="xyzw")

            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.peg_body_id])

            sensors = [square_pos, square_quat, peg_pos]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            placements = self.sampler.sample()
            for obj_pos, obj_quat, obj in placements.values():
                # Only set qpos for objects with free joints
                if hasattr(obj, "joints") and obj.joints and len(obj.joints) > 0:
                    self.sim.data.set_joint_qpos(
                        obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                    )
                else:
                    # Directly set body pose for jointless part (peg base)
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = np.array(obj_pos)
                    self.sim.model.body_quat[body_id] = np.array(obj_quat)

            # Sample and place planes
            tp_places = self.table_plane_sampler.sample()
            fp_places = self.floor_plane_sampler.sample()
            tp_pos, tp_quat, _ = tp_places[self.table_plane.name]
            fp_pos, fp_quat, _ = fp_places[self.floor_plane.name]
            self.sim.data.set_joint_qpos(
                self.table_plane.joints[0], np.concatenate([np.array(tp_pos), np.array(tp_quat)])
            )
            self.sim.data.set_joint_qpos(
                self.floor_plane.joints[0], np.concatenate([np.array(fp_pos), np.array(fp_quat)])
            )

    def _check_success(self):
        # Success if square is around peg (xy aligned and near table) and gripper is away
        square_pos = self.sim.data.body_xpos[self.square_body_id]
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]

        xy_close = (
            abs(square_pos[0] - peg_pos[0]) < 0.03
            and abs(square_pos[1] - peg_pos[1]) < 0.03
        )
        z_ok = square_pos[2] < float(self.table_offset[2]) + 0.1

        # Encourage release: require the gripper to be sufficiently far from the object
        dist_eef = min(
            [
                np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - square_pos)
                    for arm in self.robots[0].arms
                ]
            )
        r_reach = 1 - np.tanh(10.0 * dist_eef)
        gripper_away = r_reach < 0.6

        return bool(xy_close and z_ok and gripper_away)

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
