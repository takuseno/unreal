from rlsaber.util import compute_v_and_adv
from rollout import Rollout
from replay_buffer import ReplayBuffer
import build_graph
import numpy as np
import tensorflow as tf
from collections import deque


class Agent:
    def __init__(self,
                 actions,
                 optimizer,
                 convs,
                 fcs,
                 padding,
                 lstm,
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
                 shared_device='/cpu:0',
                 worker_device='/cpu:0',
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
            convs=convs,
            fcs=fcs,
            padding=padding,
            lstm=lstm,
            num_actions=len(actions),
            optimizer=optimizer,
            lstm_unit=lstm_unit,
            state_shape=state_shape,
            grad_clip=grad_clip,
            policy_factor=policy_factor,
            value_factor=value_factor,
            entropy_factor=entropy_factor,
            rp_frame=rp_frame,
            shared_device=shared_device,
            worker_device=worker_device,
            scope=name
        )

        # rnn state variables
        self.initial_state = np.zeros((1, lstm_unit), np.float32)
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state

        # last state variables
        self.zero_state = np.zeros(state_shape, dtype=np.float32)
        self.initial_last_obs = [self.zero_state for _ in range(rp_frame)]
        self.last_obs = deque(self.initial_last_obs, maxlen=rp_frame)
        self.last_action = deque([0, 0], maxlen=2)
        self.value_tm1 = None
        self.reward_tm1 = 0.0

        # buffers
        self.rollout = Rollout()
        self.buffer = ReplayBuffer(capacity=buffer_size)

        self.t = 0
        self.t_in_episode = 0

    def train(self, bootstrap_value):
        # prepare A3C update
        obs_t = np.array(self.rollout.obs_t, dtype=np.float32)
        actions_t = np.array(self.rollout.actions_t, dtype=np.uint8)
        actions_tm1 = np.array(self.rollout.actions_tm1, dtype=np.uint8)
        rewards_tp1 = self.rollout.rewards_tp1
        rewards_t = self.rollout.rewards_t
        values_t = self.rollout.values_t
        v_t, adv_t = compute_v_and_adv(
            rewards_tp1, values_t, bootstrap_value, self.gamma)
        state_t0 = self.rollout.states_t[0][0]
        state_t1 = self.rollout.states_t[0][1]

        # prepare reward prediction update
        rp_obs, rp_reward_tp1 = self.buffer.sample_rp()

        # prepare value function replay update
        vr_obs_t,\
        vr_actions_tm1,\
        vr_rewards_t,\
        is_terminal = self.buffer.sample_vr(self.time_horizon)
        _, vr_values_t, _ = self._act(vr_obs_t, vr_actions_tm1, vr_rewards_t,
                                      self.initial_state, self.initial_state,
                                      self.sess)
        vr_values_t = np.reshape(vr_values_t, [-1])
        if is_terminal:
            vr_bootstrap_value = 0.0
        else:
            vr_bootstrap_value = vr_values_t[-1]
        vr_v_t, _ = compute_v_and_adv(vr_rewards_t[:-1], vr_values_t[:-1],
                                      vr_bootstrap_value, self.gamma)

        # update
        loss = self._train(
            obs_t=obs_t,
            rnn_state0=state_t0,
            rnn_state1=state_t1,
            actions_t=actions_t,
            rewards_t=rewards_t,
            actions_tm1=actions_tm1,
            returns_t=v_t,
            advantages_t=adv_t,
            rp_obs=rp_obs,
            rp_reward_tp1=rp_reward_tp1,
            vr_obs_t=vr_obs_t[:-1],
            vr_actions_tm1=vr_actions_tm1[:-1],
            vr_rewards_t=vr_rewards_t[:-1],
            vr_returns_t=vr_v_t,
            sess=self.sess
        )
        self._update_local(self.sess)
        return loss

    def act(self, obs_t, reward_t, training=True):
        # change state shape to WHC
        obs_t = self.phi(obs_t)
        # last transitions
        action_tm2, action_tm1 = self.last_action
        obs_tm1 = self.last_obs[-1]
        # take next action
        prob, value, rnn_state = self._act(
            obs_t=[obs_t],
            actions_tm1=[action_tm1],
            rewards_t=[reward_t],
            rnn_state0=self.rnn_state0,
            rnn_state1=self.rnn_state1,
            sess=self.sess
        )
        action_t = np.random.choice(range(len(self.actions)), p=prob[0])

        if training:
            if len(self.rollout.obs_t) == self.time_horizon:
                self.train(self.value_tm1)
                self.rollout.flush()

            if self.t_in_episode > 0:
                # add transition to buffer for A3C update
                self.rollout.add(
                    obs_t=obs_tm1,
                    reward_tp1=reward_t,
                    reward_t=self.reward_tm1,
                    action_t=action_tm1,
                    action_tm1=action_tm2,
                    value_t=self.value_tm1,
                    terminal_tp1=False,
                    state_t=[self.rnn_state0, self.rnn_state1]
                )
                # add transition to buffer for auxiliary update
                self.buffer.add(
                    obs_t=list(self.last_obs),
                    action_tm1=action_tm2,
                    reward_t=self.reward_tm1,
                    action_t=action_tm1,
                    reward_tp1=reward_t,
                    obs_tp1=obs_t,
                    terminal=False
                )

        self.t += 1
        self.t_in_episode += 1
        self.rnn_state0, self.rnn_state1 = rnn_state
        self.last_obs.append(obs_t)
        self.last_action.append(action_t)
        self.value_tm1 = value[0][0]
        self.reward_tm1 = reward_t
        return self.actions[action_t]

    def stop_episode(self, obs_t, reward_t, training=True):
        # change state shape to WHC
        obs_t = self.phi(obs_t)
        # last transitions
        action_tm2, action_tm1 = self.last_action
        obs_tm1 = self.last_obs[-1]
        if training:
            # add transition for A3C update
            self.rollout.add(
                obs_t=obs_tm1,
                action_t=action_tm1,
                reward_t=self.reward_tm1,
                reward_tp1=reward_t,
                action_tm1=action_tm2,
                value_t=self.value_tm1,
                state_t=[self.rnn_state0, self.rnn_state1],
                terminal_tp1=True
            )
            # add transition for auxiliary update
            self.buffer.add(
                obs_t=list(self.last_obs),
                action_tm1=action_tm2,
                reward_t=self.reward_tm1,
                action_t=action_tm1,
                reward_tp1=reward_t,
                obs_tp1=obs_t,
                terminal=True
            )
            self.train(0.0)
            self.rollout.flush()
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = deque(self.initial_last_obs, maxlen=self.rp_frame)
        self.last_action = deque([0, 0], maxlen=2)
        self.value_tm1 = None
        self.reward_tm1 = 0.0
        self.t_in_episode = 0

    def set_session(self, sess):
        self.sess = sess
