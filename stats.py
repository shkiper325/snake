import os
import random
import json

from QNet import QNet
from env import Env
import utils

import numpy as np
from tqdm import tqdm

import cv2

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from use_cuda import *
import torch
from torch.utils.tensorboard import SummaryWriter
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

k = 4

def myzscore(x):
    mean = np.mean(x)
    std = np.std(x)

    ret = x - mean
    if std != 0:
        ret = ret / std

    return ret

def prepare_state(screen_arr):
    ret = [cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) for screen in screen_arr]

    ret = [myzscore(screen.astype(np.float32)) for screen in ret]

    ret = [np.expand_dims(screen, axis=0) for screen in ret]
    ret = np.concatenate(ret, axis=0)

    ret = np.expand_dims(ret, axis=0)

    return ret

def play(Q, env):
    frame_count = 0
    food_count = 0
    loop_count = 0

    #Playing

    env.new_game()

    hashes = set()

    last_frames = [np.full((10, 10, 3), fill_value=255, dtype=np.uint8) for i in range(k - 1)]
    last_frames.append(env.screenshot())

    while not env.finished():
        frame_count += 1

        curr_state = prepare_state(last_frames)
        
        action = None

        screen_hash = hash(curr_state.tostring())
        if screen_hash in hashes:
            loop_count += 1
            
            action = random.randint(0, 3)
        hashes.add(screen_hash)

        if action is None:
            action = FloatTensor(curr_state)
            action = Q(action)
            action = action.cpu().data.numpy()[0]
            #print(action)
            action = np.argmax(action)

        reward = env.act(action)
        if reward == 1:
            food_count += 1

        last_frames = last_frames[1:] + [env.screenshot()]

    #Return
    
    return {'food_count' : food_count, 'frame_count' : frame_count, 'loop_count' : loop_count}

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Snake DQN Agent')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode without pygame display (for servers)')
    parser.add_argument('--games', type=int, default=500, help='Number of games to evaluate per checkpoint (default: 500)')
    args = parser.parse_args()

    # Set headless mode via environment variable if specified
    if args.headless:
        import os
        os.environ['SNAKE_HEADLESS'] = '1'
        print('Running in HEADLESS mode (no display)')

    games_count = args.games

    # Initialize TensorBoard writer for evaluation
    writer = SummaryWriter(log_dir='./runs/evaluation')

    #Engine init
    env = Env({
        'border_width' : 50,
        'square_size' : 20,
        'snake_color' : np.array([90, 0, 157]),
        'food_color' : np.array([255, 0, 0]),
        'background_color' : np.array([255, 255, 255]),
        'border_background_color' : np.array([255, 255, 255]),
        'border_line_color' : np.array([0, 0, 0]),
        'snake_head_color' : np.array([0, 255, 0]),
        'border_line_width' : 3,
        'field_size' : (8, 8),
        'snake_init_len' : 4,
        'food_count' : 1,
        'food_score' : 1,
        'death_score' : -1,
        'survive_score' : 0,
        'torus' : False,
        'headless' : args.headless
    })

    #Load names
    models_dir = 'models/'

    states = [int(name) for name in os.listdir(models_dir) if name not in ['state', 'state_bak']]
    states.sort()

    mean_frames = []
    mean_food = []
    mean_loops_ratio = []
    mean_ff_ratio = []

    Q = QNet(4, 4)

    for state in tqdm(states):
        weights = torch.load(os.path.join(models_dir, str(state)))['Q_targ']
        Q.load_state_dict(weights)

        frames = []
        food = []
        loops_ratio = []
        ff_ratio = []

        for i in range(games_count):
            res = play(Q, env)

            frames.append(res['frame_count'])
            food.append(res['food_count'])
            loops_ratio.append(res['loop_count'] / res['frame_count'])
            ff_ratio.append(res['food_count'] / res['frame_count'])

        mean_frames.append(np.mean(frames))
        mean_food.append(np.mean(food))
        mean_loops_ratio.append(np.mean(loops_ratio))
        mean_ff_ratio.append(np.mean(ff_ratio))

        # Log to TensorBoard
        writer.add_scalar('Evaluation/MeanFramesCount', np.mean(frames), state)
        writer.add_scalar('Evaluation/MeanFoodCount', np.mean(food), state)
        writer.add_scalar('Evaluation/MeanLoopsRatio', np.mean(loops_ratio), state)
        writer.add_scalar('Evaluation/MeanFoodPerFrame', np.mean(ff_ratio), state)

    # Close TensorBoard writer
    writer.close()

    # PNG plots disabled - use TensorBoard instead
    # plt.close()
    # plt.figure()
    # plt.plot(states, mean_frames, 'o')
    # plt.xlabel('Frames seen')
    # plt.ylabel('Mean frames count')
    # plt.savefig('frames.png')

    # plt.close()
    # plt.figure()
    # plt.plot(states, mean_food, 'r')
    # plt.xlabel('Frames seen')
    # plt.ylabel('Mean food count')
    # plt.savefig('food.png')

    # plt.close()
    # plt.figure()
    # plt.plot(states, mean_loops_ratio, 'g')
    # plt.xlabel('Frames seen')
    # plt.ylabel('Mean loops/frames ratio')
    # plt.savefig('loops.png')

    # plt.close()
    # plt.figure()
    # plt.plot(states, mean_ff_ratio, 'b')
    # plt.xlabel('Frames seen')
    # plt.ylabel('Mean food/frames ratio')
    # plt.savefig('ff_ratio.png')

    out = {
        'states' : states,
        'mean_frames' : mean_frames,
        'mean_food' : mean_food,
        'mean_loops_ratio' : mean_loops_ratio,
        'mean_ff_ratio' : mean_ff_ratio
    }

    json_str = json.dumps(out)
    fd = open('result.json', 'w')
    fd.write(json_str)
    fd.close()