import os
import shutil

import torch
import torch.optim as optim

from QNet import QNet
from replay_memory import ReplayMemory
import utils

if os.path.exists('models'):
    shutil.rmtree('models')
os.mkdir('models')
if os.path.exists('states'):
    shutil.rmtree('states')
os.mkdir('states')

Q = QNet(num_actions=4, k=4)
Q.apply(utils.init_weights)

torch.save(
    {
        'Q' : Q.state_dict(),
        'Q_targ' : Q.state_dict()
    },
    'models/0'
)

optimizer = optim.Adam(Q.parameters(), lr=0.0000625, eps=1.5e-4)
replay_mem = ReplayMemory(max_size=500000, alpha=0.5, eps=0.0)

torch.save(
    {
        'replay_mem' : replay_mem,
        'end_eps' : 1,
        'end_episode' : 0,
        'curr_frame_count' : 0,
        'optim' : optimizer.state_dict()
    },
    'states/0'
)