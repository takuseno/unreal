# DeepMind style
STATE_SHAPE = [84, 84]
STATE_WINDOW = 1
CONVS = [[16, 8, 4], [32, 4, 2]]
PADDING = 'VALID'
FCS = [256]
OPTIMIZER = 'rmsprop'
# OpenAI syle
#STATE_SHAPE = [42, 42]
#STATE_WINDOW = 1
#CONVS = [[32, 3, 2], [32, 3, 2], [32, 3, 2], [32, 3, 2]]
#PADDING = 'SAME'
#FCS = []
#OPTIMIZER = 'adam'

LSTM_UNIT = 256
FINAL_STEP = 10 ** 8

LSTM = True
POLICY_FACTOR = 1.0
VALUE_FACTOR = 0.5
ENTROPY_FACTOR = 0.001
LR = 7e-4
LR_DECAY = 'linear'
GRAD_CLIP = 40.0
TIME_HORIZON = 20
GAMMA = 0.99
RP_FRAME = 3
BUFFER_SIZE = 2e3

EVAL_INTERVAL = 10 ** 6
EVAL_EPISODES = 5
RECORD_EPISODES = 3
