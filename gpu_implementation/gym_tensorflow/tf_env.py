__copyright__ = """
Copyright (c) 2018 Uber Technologies, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np
import os
import tensorflow as tf
from vizdoom import *
import cv2

gym_tensorflow_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'gym_tensorflow.so'))


class TensorFlowEnv(object):
    pass


class PythonEnv(TensorFlowEnv):
    def step(self, action, indices=None, name=None):
        with tf.variable_scope(name, default_name='PythonStep'):
            reward, done = tf.py_func(self._step, [action, indices], [tf.float32, tf.bool])
            reward.set_shape(indices.get_shape())
            done.set_shape(indices.get_shape())
            return reward, done

    def _reset(self, indices):
        raise NotImplementedError()

    def reset(self, indices=None, max_frames=None, name=None):
        with tf.variable_scope(name, default_name='PythonReset'):
            return tf.py_func(self._reset, [indices], tf.int64).op

    def _step(self, action, indices):
        raise NotImplementedError()

    def _obs(self, indices):
        raise NotImplementedError()

    def observation(self, indices=None, name=None):
        with tf.variable_scope(name, default_name='PythonObservation'):
            obs = tf.py_func(self._obs, [indices], tf.float32)
            obs.set_shape(tuple(indices.get_shape()) + self.observation_space)
            return tf.expand_dims(obs, axis=1)

    def final_state(self, indices, name=None):
        with tf.variable_scope(name, default_name='PythonFinalState'):
            return tf.zeros([tf.shape(indices)[0], 2], dtype=tf.float32)

    @property
    def unwrapped(self):
        return self

    def close(self):
        pass


class GymEnv(PythonEnv):
    def __init__(self, name, batch_size):
        import gym
        self.env = [gym.make(name) for _ in range(batch_size)]
        self.obs = [None] * batch_size

    @property
    def env_default_timestep_cutoff(self):
        return 100000

    @property
    def action_space(self):
        return np.prod(self.env[0].action_space.shape)

    @property
    def observation_space(self):
        return self.env[0].observation_space.shape

    @property
    def discrete_action(self):
        return False

    def _step(self, action, indices):
        assert self.discrete_action == False
        results = map(lambda i: self.env[indices[i]].step(action[i]), range(len(indices)))
        obs, reward, done, _ = zip(*results)
        for i in range(len(indices)):
            self.obs[indices[i]] = obs[i].astype(np.float32)

        return np.array(reward, dtype=np.float32), np.array(done, dtype=np.bool)

    def _reset(self, indices):
        for i in indices:
            self.obs[i] = self.env[i].reset().astype(np.float32)
        return 0

    def _obs(self, indices):
        return np.array([self.obs[i] for i in indices]).astype(np.float32)


class DoomEnv(PythonEnv):
    def __init__(self, name, batch_size):
        self.env = [DoomEnvBase(name) for _ in range(batch_size)]
        self.obs = [None] * batch_size
        self.env_default_timestep_cutoff = 5000

    @property
    def action_space(self):
        return self.env[0].num_actions

    @property
    def observation_space(self):
        return self.env[0].observation_space

    @property
    def discrete_action(self):
        return True

    def step(self, action, indices=None, name=None):
        if indices is None:
            indices = [i for i in range(len(self.obs))]

        with tf.variable_scope(name, default_name='PythonStep'):
            reward, done = tf.py_func(self._step, [action, indices], [tf.float32, tf.bool])

            reward = tf.convert_to_tensor(reward)
            done = tf.convert_to_tensor(done)

            reward.set_shape((None, 1))
            done.set_shape((None, 1))

            # reward.set_shape(indices.get_shape())
            # done.set_shape(indices.get_shape())
            return reward, done

    def reset(self, indices=None, max_frames=None, name=None):
        if indices is None:
            indices = [i for i in range(len(self.obs))]

        with tf.variable_scope(name, default_name='PythonReset'):
            return tf.py_func(self._reset, [indices], tf.int64).op

    def observation(self, indices=None, name=None):
        if indices is None:
            indices = [i for i in range(len(self.obs))]

        with tf.variable_scope(name, default_name='PythonObservation'):
            obs = tf.py_func(self._obs, [indices], tf.float32)
            obs = tf.convert_to_tensor(obs)
            obs.set_shape((None, 84, 84, 4))
            return obs
            # obs.set_shape((None, ) + (84, 84) + (1,))
            # obs.set_shape(tuple(indices.get_shape()) + self.observation_space)
            # return tf.expand_dims(obs, axis=1)


    def final_state(self, indices, name=None):
        with tf.variable_scope(name, default_name='PythonFinalState'):
            return tf.zeros([tf.shape(indices)[0], 2], dtype=tf.float32)

    def _step(self, action, indices):
        results = map(lambda i: self.env[indices[i]].step(action[i]), range(len(indices)))
        obs, reward, done, _ = zip(*results)
        for i in range(len(indices)):
            self.obs[indices[i]] = obs[i].astype(np.float32)
        return np.array(reward, dtype=np.float32), np.array(done, dtype=np.bool)


    def _reset(self, indices=None):
        for i in indices:
            self.obs[i] = self.env[i].reset().astype(np.float32)
        return 0

    def _obs(self, indices):
        return np.array([self.obs[i] for i in indices]).astype(np.float32)

    def close(self):
        game.close()


class DoomEnvBase:
    def __init__(self, name, visible=False):
        game = DoomGame()

        if name == 'doomtakecover':
            cfg_path = '/home/bzhou/tmp/ViZDoom/scenarios/take_cover.cfg'
            game.load_config(cfg_path)
            self.actions = [[True, False], [False, True], [False, False]]
        else:
            cfg_path = '/home/bzhou/tmp/ViZDoom/scenarios/health_gathering.cfg'
            game.load_config(cfg_path)
            self.actions = [[True, False, False], [False, True, False], [False, False, True]]

        if visible:
            game.set_screen_resolution(ScreenResolution.RES_640X480)
        else:
            game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.BGR24)
        game.set_window_visible(visible)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.game = game
        self.num_actions = len(self.actions)
        self.observation_space = (84, 84, 3)
        self.discrete_action = self.actions
        self.screen_buff = []


    def preprocess(self, img):
        img = cv2.resize(img, (84, 84))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def preprocess_(self, obs):
        obs = obs.astype(np.float32) / 255.0
        obs = np.array(resize(obs, (84, 84)))
        obs = ((1.0 - obs) * 255).round().astype(np.uint8)
        return obs

    def reset(self):
        self.game.new_episode()
        img = self.game.get_state().screen_buffer
        img = self.preprocess(img)
        self.screen_buff = [img] * 4
        return np.array(self.screen_buff).transpose(1, 2, 0)

    def step(self, action):
        action = self.actions[action]
        reward = self.game.make_action(action)
        done = self.game.is_episode_finished()
        if not done:
            img = self.game.get_state().screen_buffer
            img = self.preprocess(img)
            self.screen_buff.append(img)
            self.screen_buff.pop(0)

        return np.array(self.screen_buff).transpose(1, 2, 0), reward, done, None


if __name__ == '__main__':
    sess = tf.Session()
    # env = GymEnv('Breakout-v0', 5)
    env = DoomEnv('doomtakecover', 5)
    obs = env.reset()
    print(sess.run(obs))



