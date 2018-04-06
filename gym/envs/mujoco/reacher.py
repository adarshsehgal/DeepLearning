import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'aubo_i5.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("right_gripper_link")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        if (abs(reward_dist) < 0.05):
            done = True
        else:
            done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 13
        self.viewer.cam.elevation = -40
        self.viewer.cam.distance = self.model.stat.extent * 1.5

    def reset_model(self):
        qpos = self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.1, high=.1, size=3)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-3:] = self.goal
        qvel = self.init_qvel
        qvel[-3:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("right_gripper_link") - self.get_body_com("target")
        ])
