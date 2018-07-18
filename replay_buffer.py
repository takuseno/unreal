from collections import deque
from random import sample, randrange, random
import uuid


class ReplayBuffer:
    def __init__(self, capacity=2e3):
        self.capacity = capacity
        self.ids = []
        self.transitions = {}
        self.rewarding_states = {}
        self.non_rewarding_states = {}
        self.episode_terminal_ids = []

    # ((s_t-2, s_t-1, s_t), a_t-1, r_t, a_t, r_t+1, s_t+1, t_t+1)
    def add(self, states, action_tm1, reward_t, action_t, reward_tp1, state_tp1, terminal):
        # create unique id
        id = uuid.uuid4()
        self.ids.append(id)

        # remove oldest transision
        if len(self.transitions.keys()) > self.capacity:
            self.remove(self.ids[0])

        # for value function replay and others
        transition = dict(
            state_t=states[-1],
            action_tm1=action_tm1,
            reward_t=reward_t,
            action_t=action_t,
            reward_tp1=reward_tp1,
            state_tp1=state_tp1
        )
        self.transitions[id] = transition

        # for reward prediction
        reward_prediction_dict = dict(states=states, reward_tp1=reward_tp1)
        if reward_tp1 == 0.0:
            self.non_rewarding_states[id] = reward_prediction_dict
        else:
            self.rewarding_states[id] = reward_prediction_dict

        # add episode terminal id
        if terminal:
            self.episode_terminal_ids.append(id)

    def remove(self, id):
        if id in self.ids:
            self.ids.remove(id)
            self.transitions.pop(id)
            if id in self.episode_terminal_ids:
                self.episode_terminal_ids.remove(id)
            if id in self.rewarding_states:
                self.rewarding_states.pop(id)
            if id in self.non_rewarding_states:
                self.non_rewarding_states.pop(id)

    def sample_rp(self):
        prob = random()
        if prob > 0.5 and len(self.rewarding_states.values()) != 0:
            transition = sample(list(self.rewarding_states.values()), 1)[0]
        else:
            transition = sample(list(self.non_rewarding_states.values()), 1)[0]
        reward = transition['reward_tp1']
        if reward == 0.0:
            reward_class = 0
        elif reward > 0.0:
            reward_class = 1
        else:
            reward_class = 2
        return transition['states'], reward_class

    def sample_sequence(self, n):
        if len(self.episode_terminal_ids) > 0:
            # get terminal index
            episode_index = randrange(len(self.episode_terminal_ids))
            id = self.episode_terminal_ids[episode_index]
            end_index = self.ids.index(id)

            # get start index
            if episode_index == 0:
                start_index = 0
            else:
                prev_id = self.episode_terminal_ids[episode_index - 1]
                start_index = self.ids.index(prev_id) + 1
        else:
            # no episode ends yet
            end_index = len(self.ids) - 1
            start_index = 0

        # get trajectory
        length = end_index - start_index + 1
        if length > n:
            sample_start = randrange(length - n + 1) + start_index
            sample_end = sample_start + n - 1
        else:
            sample_start = start_index
            sample_end = end_index
        transitions = list(self.transitions.values())[sample_start:sample_end+1]
        is_terminal = self.ids[sample_end] in self.episode_terminal_ids
        return transitions, is_terminal

    def sample_vr(self, n):
        transitions, is_terminal = self.sample_sequence(n)
        # format results
        states_t = []
        actions_tm1 = []
        rewards_t = []
        for transition in transitions:
            states_t.append(transition['state_t'])
            actions_tm1.append(transition['action_tm1'])
            rewards_t.append(transition['reward_t'])
        states_t.append(transitions[-1]['state_tp1'])
        actions_tm1.append(transitions[-1]['action_t'])
        rewards_t.append(transitions[-1]['reward_tp1'])
        return states_t, actions_tm1, rewards_t, is_terminal
