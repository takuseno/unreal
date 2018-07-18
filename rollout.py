class Rollout:
    def __init__(self):
        self.flush()

    def add(self, state, action, reward, last_action,
            value, terminal=False, feature=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.last_actions.append(last_action)
        self.values.append(value)
        self.terminals.append(terminal)
        self.features.append(feature)

    def flush(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.last_actions = []
        self.values = []
        self.terminals = []
        self.features = []
