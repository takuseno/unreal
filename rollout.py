class Rollout:
    def __init__(self):
        self.flush()

    def add(self, obs_t, action_t, reward_t, reward_tp1, action_tm1,
            value_t, terminal_tp1=False, state_t=None):
        self.obs_t.append(obs_t)
        self.actions_t.append(action_t)
        self.rewards_t.append(reward_t)
        self.rewards_tp1.append(reward_tp1)
        self.actions_tm1.append(action_tm1)
        self.values_t.append(value_t)
        self.terminals_tp1.append(terminal_tp1)
        self.states_t.append(state_t)

    def flush(self):
        self.obs_t = []
        self.actions_t = []
        self.rewards_t = []
        self.rewards_tp1 = []
        self.actions_tm1 = []
        self.values_t = []
        self.terminals_tp1 = []
        self.states_t = []
