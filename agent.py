from rlsaber.util import compute_v_and_adv
from rollout import Rollout
from replay_buffer import ReplayBuffer
import build_graph
import numpy as np
import tensorflow as tf
from collections import deque


class Agent:
    def __init__(self,
                 model,
                 actions,
                 optimizer,
                 gamma=0.99,
                 lstm_unit=256,
                 time_horizon=5,
                 policy_factor=1.0,
                 value_factor=0.5,
                 entropy_factor=0.01,
                 grad_clip=40.0,
                 state_shape=[84, 84, 1],
                 buffer_size=2e3,
                 rp_frame=3,
                 phi=lambda s: s,
                 name='global',
                 sess=None):
        self.actions = actions
        self.gamma = gamma
        self.name = name
        self.time_horizon = time_horizon
        self.state_shape = state_shape
        self.rp_frame = rp_frame
        self.phi = phi
        self.sess = sess

        self._act,\
        self._train,\
        self._update_local = build_graph.build_train(
            model=model,
            num_actions=len(actions),
            optimizer=optimizer,
            lstm_unit=lstm_unit,
            state_shape=state_shape,
            grad_clip=grad_clip,
            policy_factor=policy_factor,
            value_factor=value_factor,
            entropy_factor=entropy_factor,
            scope=name
        )

        # rnn state variables
        self.initial_state = np.zeros((1, lstm_unit), np.float32)
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state

        # last state variables
        self.last_obs = deque([
            np.zeros(state_shape, dtype=np.float32)], maxlen=rp_frame)
        self.last_action = deque([0, 0], maxlen=2)
        self.last_value = None

        # buffers
        self.rollout = Rollout()
        self.buffer = ReplayBuffer(capacity=buffer_size)

        self.t = 0
        self.t_in_episode = 0

    def train(self, bootstrap_value, reward):
        states = np.array(self.rollout.states, dtype=np.float32)
        actions = np.array(self.rollout.actions, dtype=np.uint8)
        last_actions = np.array(self.rollout.last_actions, dtype=np.uint8)
        rewards = self.rollout.rewards + [reward]
        values = self.rollout.values
        v, adv = compute_v_and_adv(rewards[:-1], values, bootstrap_value, self.gamma)
        loss = self._train(
            states, self.rollout.features[0][0], self.rollout.features[0][1],
            actions, rewards[1:], last_actions, v, adv, self.sess)
        self._update_local(self.sess)
        return loss

    def act(self, obs, reward, training=True):
        # change state shape to WHC
        obs = self.phi(obs)
        # take next action
        prob, value, rnn_state = self._act(
            [obs], [self.last_action[0]], [reward],
            self.rnn_state0, self.rnn_state1, self.sess)
        action = np.random.choice(range(len(self.actions)), p=prob[0])

        if training:
            if len(self.rollout.states) == self.time_horizon:
                self.train(self.last_value, reward)
                self.rollout.flush()

            if self.t_in_episode > 0:
                # add transition to buffer for A3C update
                self.rollout.add(
                    state=self.last_obs[-1],
                    reward=reward,
                    action=self.last_action[1],
                    last_action=self.last_action[0],
                    value=self.last_value,
                    terminal=False,
                    feature=[self.rnn_state0, self.rnn_state1]
                )
                # add transition to buffer for auxiliary update
                self.buffer.add(
                    states=list(self.last_obs),
                    reward=reward,
                    next_state=obs,
                    terminal=False
                )

        self.t += 1
        self.t_in_episode += 1
        self.rnn_state0, self.rnn_state1 = rnn_state
        self.last_obs.append(obs)
        self.last_action.append(action)
        self.last_value = value[0][0]
        return self.actions[action]

    def stop_episode(self, obs, reward, training=True):
        if training:
            # add transition for A3C update
            self.rollout.add(
                state=self.last_obs[-1],
                action=self.last_action[1],
                reward=reward,
                last_action=self.last_action[0],
                value=self.last_value,
                feature=[self.rnn_state0, self.rnn_state1],
                terminal=True
            )
            # add transition for auxiliary update
            self.buffer.add(
                states=list(self.last_obs),
                reward=reward,
                next_state=obs,
                terminal=True
            )
            self.train(0, 0.0)
            self.rollout.flush()
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = deque([np.zeros(self.state_shape, dtype=np.float32)],
                              maxlen=self.rp_frame)
        self.last_action = deque([0, 0], maxlen=2)
        self.last_value = None
        self.t_in_episode = 0

    def set_session(self, sess):
        self.sess = sess
