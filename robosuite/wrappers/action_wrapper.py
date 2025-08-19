import numpy as np
import robosuite.utils.transform_utils as T


EEF_SITE_NAME = "gripper0_right_grip_site"


def wrap_env_action_space(env, action_space: str):
    """Dynamically subclass env to support delta / absolute action spaces.

    Call like:
        env = wrap_env_action_space(env, "eef_delta")
        env.set_init_action()
        env.step(action)
    """

    class WrappedEnv(env.__class__):
        def set_init_action(self):
            if action_space.startswith("eef_"):
                site_id = self.sim.model.site_name2id(EEF_SITE_NAME)
                pos = self.sim.data.site_xpos[site_id].copy()
                rot = self.sim.data.site_xmat[site_id].reshape(3, 3)
                aa = T.quat2axisangle(T.mat2quat(rot))
                self._asw_last_abs_pose = np.concatenate([pos, aa])
            elif action_space.startswith("joint_"):
                self._asw_last_joint_pos = self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes].copy()

        def step(self, action: np.ndarray):
            if action_space == "eef_abs":
                return super(WrappedEnv, self).step(action)
            if action_space == "eef_delta":
                if self._asw_last_abs_pose is None:
                    raise ValueError("Call set_init_action() before stepping with eef_delta actions")
                delta_pose = action[0:6]
                self._asw_last_abs_pose = self._asw_last_abs_pose + delta_pose
                abs_action = np.concatenate([self._asw_last_abs_pose, action[-1:]])
                return super(WrappedEnv, self).step(abs_action)
            if action_space == "joint_abs":
                return super(WrappedEnv, self).step(action)
            if action_space == "joint_delta":
                if self._asw_last_joint_pos is None:
                    raise ValueError("Call set_init_action() before stepping with joint_delta actions")
                delta_joints = action[:-1]
                self._asw_last_joint_pos = self._asw_last_joint_pos + delta_joints
                abs_action = np.concatenate([self._asw_last_joint_pos, action[-1:]])
                return super(WrappedEnv, self).step(abs_action)
            raise ValueError(f"Unsupported action_space: {action_space}")

    env.__class__ = WrappedEnv
    env._asw_action_space = action_space
    env._asw_last_abs_pose = None
    env._asw_last_joint_pos = None
    return env


