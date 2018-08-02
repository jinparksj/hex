import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os


class SixLeggedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = os.path.split(os.path.realpath(__file__))[0]+"/silvia1.xml"
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        zposbefore = self.get_body_com("torso")[2]

        self.do_simulation(a, self.frame_skip)

        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        zposafter = self.get_body_com("torso")[2]

        # Moves in negative y direction
        forward_reward = -(yposbefore - yposafter)/self.dt

        ctrl_cost = .01 * np.square(a).sum()
        contact_cost = 0.01 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.1
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            # first 7 qpos is for translation, rotation, and 4th quaternion term
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def get_actuator_pos0(self):
        return self.sim.model.qpos0.flat

    def get_actuator_pos(self):
        return self.sim.data.qpos.flat

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
