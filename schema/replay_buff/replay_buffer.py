from schema.misc.serializable import Serializable
import numpy as np
import abc


class ReplayBuffer(object):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def size(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by railrl.samplers.util.rollout
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path.get("agent_infos", {}),
            path.get("env_infos", {}),
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass


class SimpleReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, env_spec, max_replay_buffer_size):
        super(SimpleReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        max_replay_buffer_size = int(max_replay_buffer_size)

        self._env_spec = env_spec
        self._observation_dim = env_spec.observation_flat_dim
        self._action_dim = env_spec.action_flat_dim
        self._max_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros(max_replay_buffer_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(max_replay_buffer_size, dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return {
            'observations': self._observations[indices],
            'actions': self._actions[indices],
            'rewards': self._rewards[indices],
            'terminals': self._terminals[indices],
            'next_observations': self._next_obs[indices]
        }

    @property
    def size(self):
        return self._size

    def __getstate__(self):
        buffer_state = super(SimpleReplayBuffer, self).__getstate__()
        buffer_state.update({
            'observations': self._observations.tobytes(),
            'actions': self._actions.tobytes(),
            'rewards': self._rewards.tobytes(),
            'terminals': self._terminals.tobytes(),
            'next_observations': self._next_obs.tobytes(),
            'top': self._top,
            'size': self._size,
        })
        return buffer_state

    def __setstate__(self, buffer_state):
        super(SimpleReplayBuffer, self).__setstate__(buffer_state)

        flat_obs = np.fromstring(buffer_state['observations'])
        flat_next_obs = np.fromstring(buffer_state['next_observations'])
        flat_actions = np.fromstring(buffer_state['actions'])
        flat_reward = np.fromstring(buffer_state['rewards'])
        flat_terminals = np.fromstring(
            buffer_state['terminals'], dtype=np.uint8)

        self._observations = flat_obs.reshape(self._max_buffer_size, -1)
        self._next_obs = flat_next_obs.reshape(self._max_buffer_size, -1)
        self._actions = flat_actions.reshape(self._max_buffer_size, -1)
        self._rewards = flat_reward.reshape(self._max_buffer_size)
        self._terminals = flat_terminals.reshape(self._max_buffer_size)
        self._top = buffer_state['top']
        self._size = buffer_state['size']


class UnionBuffer(ReplayBuffer):
    def __init__(self, buffers):
        buffer_sizes = np.array([b.size for b in buffers])
        self._total_size = sum(buffer_sizes)
        self._normalized_buffer_sizes = buffer_sizes / self._total_size

        self.buffers = buffers

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError

    def terminate_episode(self):
        raise NotImplementedError

    @property
    def size(self):
        return self._total_size

    def add_path(self, **kwargs):
        raise NotImplementedError

    def random_batch(self, batch_size):

        # TODO: Hack
        partial_batch_sizes = self._normalized_buffer_sizes * batch_size
        partial_batch_sizes = partial_batch_sizes.astype(int)
        partial_batch_sizes[0] = batch_size - sum(partial_batch_sizes[1:])

        partial_batches = [
            buffer.random_batch(partial_batch_size) for buffer,
            partial_batch_size in zip(self.buffers, partial_batch_sizes)
        ]

        def all_values(key):
            return [partial_batch[key] for partial_batch in partial_batches]

        keys = partial_batches[0].keys()

        return {key: np.concatenate(all_values(key), axis=0) for key in keys}

