import time
import random
import os
import sys
import math
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import cv2

from QNet import QNet
from replay_memory import ReplayMemory
from env import Env
import utils

from use_cuda import *
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
BoolTensor = torch.cuda.BoolTensor if USE_CUDA else torch.BoolTensor

if USE_CUDA:
    print('Using CUDA')
else:
    print('Using CPU')

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

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Snake DQN Agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--frame-count', type=int, default=7008000, help='Total training frames (default: 7008000)')
    parser.add_argument('--eps-decay-time', type=float, default=0.5, help='Epsilon decay time fraction (default: 0.5)')
    parser.add_argument('--episode-depth', type=int, default=10000, help='Max steps per episode (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--eps-start', type=float, default=1.0, help='Initial epsilon (default: 1.0)')
    parser.add_argument('--eps-end', type=float, default=0.05, help='Final epsilon (default: 0.05)')
    parser.add_argument('--q-targ-update-freq', type=int, default=32000, help='Target network update frequency (default: 32000)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode without pygame display (for servers)')
    args = parser.parse_args()

    # Set headless mode via environment variable if specified
    if args.headless:
        os.environ['SNAKE_HEADLESS'] = '1'
        print('Running in HEADLESS mode (no display)')

    #Save/load dirs
    models_dir = './models'
    states_dir = './states'
    logs_dir = './runs'

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    if not os.path.exists(states_dir):
        os.mkdir(states_dir)
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=logs_dir)

    #Engine parameters
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

    num_actions = 4

    #Last frames count definition
    k = 4

    #Initing neural networks and loading previos state, if exists
    Q = QNet(num_actions=num_actions, k=k)
    Q_targ = QNet(num_actions=num_actions, k=k)

    prev_state = None
    if os.listdir(models_dir) == []:
        print('Cold start')

        Q.apply(utils.init_weights)
        Q_targ.load_state_dict(Q.state_dict())
    else:
        print('Loading weights')

        weights_file_path, state_file_path = utils.find_prev_state_files(models_dir, states_dir)

        weights = torch.load(weights_file_path)
        prev_state = torch.load(state_file_path)

        Q.load_state_dict(weights['Q'])
        Q_targ.load_state_dict(weights['Q_targ'])

    #Learn params (from command line arguments)
    gamma = args.gamma

    #Hyperparams (from command line arguments)
    frame_count = args.frame_count
    eps_decay_time = args.eps_decay_time
    episode_depth = args.episode_depth
    batch_size = args.batch_size
    eps_start = args.eps_start
    eps_end = args.eps_end
    Q_targ_update_freq = args.q_targ_update_freq

    # Print hyperparameters
    print('Hyperparameters:')
    print(f'  gamma: {gamma}')
    print(f'  frame_count: {frame_count}')
    print(f'  eps_decay_time: {eps_decay_time}')
    print(f'  episode_depth: {episode_depth}')
    print(f'  batch_size: {batch_size}')
    print(f'  eps_start: {eps_start}')
    print(f'  eps_end: {eps_end}')
    print(f'  Q_targ_update_freq: {Q_targ_update_freq}')
    print()

    # Log all hyperparameters to TensorBoard
    hparams = {
        'gamma': gamma,
        'frame_count': frame_count,
        'eps_decay_time': eps_decay_time,
        'episode_depth': episode_depth,
        'batch_size': batch_size,
        'eps_start': eps_start,
        'eps_end': eps_end,
        'Q_targ_update_freq': Q_targ_update_freq,
        'learning_rate': 0.0000625,
        'optimizer_eps': 1.5e-4,
        'replay_memory_size': 500000,
        'replay_memory_alpha': 0.5,
        'save_frequency': frame_count // 100,
    }

    # Add text summary with all hyperparameters
    hparam_text = '\n'.join([f'{k}: {v}' for k, v in hparams.items()])
    writer.add_text('Hyperparameters/All', hparam_text, 0)

    # Log environment parameters
    env_params_text = f'''
Environment Configuration:
- Field size: {env.field_size[0]}x{env.field_size[1]}
- Snake initial length: {env.snake_init_len}
- Food count: {env.food_count}
- Food score: {env.food_score}
- Death score: {env.death_score}
- Survive score: {env.survive_score}
- Torus mode: {env.torus}
- Square size: {env.square_size}
- Border width: {env.border_width}
'''
    writer.add_text('Environment/Configuration', env_params_text, 0)

    #Useful variables
    l = -math.log(eps_end) / (frame_count * eps_decay_time)

    replay_mem = ReplayMemory(max_size=500000, alpha=0.5, eps=0.0) if prev_state is None else prev_state['replay_mem']

    save_freq = frame_count // 100

    #Episode loop
    curr_eps = eps_start if prev_state is None else prev_state['end_eps']

    episode_num = 0 if prev_state is None else (prev_state['end_episode'] + 1)
    curr_frame_count = 0 if prev_state is None else prev_state['curr_frame_count']

    #Optimizer init
    if prev_state is None:
        optimizer = torch.optim.Adam(Q.parameters(), lr=0.0000625, eps=1.5e-4)
    else:
        optimizer = torch.optim.Adam(Q.parameters(), lr=0.0000625, eps=1.5e-4)
        optimizer.load_state_dict(prev_state['optim'])
    
    last_curr_frame_count = curr_frame_count
    last_time = time.time()
    avg_speed = None

    # Track episode stats for TensorBoard
    episode_food_count = 0
    episode_frame_count = 0

    # Track final metrics for hparams summary
    final_metrics = {
        'final_food_count': 0,
        'final_frame_count': 0,
        'final_food_per_frame': 0,
        'final_loss': 0
    }

    while True:
        if curr_frame_count > frame_count:
            break

        print()
        print('===================================================')
        print('Episode number:', episode_num)
        print('Frames processed:', round(curr_frame_count / frame_count * 100, 2), '%')

        losses = []
        episode_food_count = 0
        episode_frame_count = 0

        env.new_game()

        last_frames = [np.full((10, 10, 3), fill_value=255, dtype=np.uint8) for i in range(k - 1)]
        last_frames.append(env.screenshot())

        curr_state = prepare_state(last_frames)

        for t in range(episode_depth):
            #
            # Acting
            #

            if env.finished():
                break

            print(t, ' ', end='')
            sys.stdout.flush()

            Q_out = Q(FloatTensor(curr_state)).data.cpu().numpy()[0]

            # action = None
            # if random.random() < curr_eps:
            #     if curr_frame_count <= frame_count // 2:
            #         action = random.randint(0, num_actions - 1)
            #     else:
            #         action = np.argmax(Q_out)
            # else:
            #     action = np.argmax(Q_out)

            action = None
            if random.random() < curr_eps:
                action = random.randint(0, num_actions - 1)
            else:
                action = np.argmax(Q_out)

            reward = env.act(action)

            # Track episode stats
            episode_frame_count += 1
            if reward > 0:  # Food eaten
                episode_food_count += 1

            loss = reward - np.amax(Q_out)

            new_state = None
            if not env.finished():
                last_frames = last_frames[1:] + [env.screenshot()]
                new_state = prepare_state(last_frames)

                Q_out = Q(FloatTensor(new_state)).detach().cpu().numpy()[0]
                Q_targ_out = Q_targ(FloatTensor(new_state)).detach().cpu().numpy()[0]

                loss += gamma * Q_targ_out[np.argmax(Q_out)]

            loss = abs(loss)

            replay_mem.add_element((curr_state, action, reward, new_state), loss)

            curr_state = new_state

            #
            #Learning
            #

            sarses = replay_mem.get_batch(batch_size)

            #Targets

            Q_true = []
            if any([sars[3] is not None for sars in sarses]):
                batch = np.concatenate([sars[3] for sars in sarses if sars[3] is not None], axis=0)
                batch = FloatTensor(batch)

                actions = np.argmax(Q(batch).data.cpu().numpy(), axis=1)
                Q_targ_out = Q_targ(batch).data.cpu().numpy()
                Q_true = [Q_targ_out[i][actions[i]] for i in range(Q_targ_out.shape[0])]
            
            j = 0
            Q_true_new = []
            for i in range(len(sarses)):
                if sarses[i][3] is None:
                    Q_true_new.append(0)
                else:
                    Q_true_new.append(Q_true[j])
                    j += 1
            Q_true = np.array(Q_true_new)

            rs = np.array([sars[2] for sars in sarses])
            Q_true = rs + gamma * Q_true
            
            #Predictions

            batch = np.concatenate([sars[0] for sars in sarses], axis=0)
            batch = FloatTensor(batch)
            Q_pred = Q(batch)
            mask = np.array([utils.dirac_delta(i=sars[1], n=num_actions) for sars in sarses], dtype=bool)
            mask = BoolTensor(mask)
            Q_pred = torch.masked_select(Q_pred, mask=mask)

            #Updating replay memory

            replay_mem.update(np.abs(Q_true - Q_pred.detach().cpu().numpy()))

            #Learning

            loss = nn.SmoothL1Loss()(Q_pred, FloatTensor(Q_true))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.data.cpu().numpy()))

            #
            #Updating frame counter
            #

            curr_frame_count += 1

            if curr_frame_count - last_curr_frame_count >= 500:
                last_curr_frame_count = curr_frame_count
                avg_speed = 500 / (time.time() - last_time)
                last_time = time.time()

            #
            #Updating eps
            #

            curr_eps = max(math.exp(-l * curr_frame_count), eps_end)

            #
            #Updating Q_targ if needed
            #

            if curr_frame_count % Q_targ_update_freq == 0:
                print('Updating Q_targ')

                Q_targ.load_state_dict(Q.state_dict())

            #
            #Saving if needed
            #

            if curr_frame_count % save_freq == 0:
                print('Saving model')

                torch.save({
                    'Q' : Q.state_dict(),
                    'Q_targ' : Q_targ.state_dict()
                }, os.path.join(models_dir, str(curr_frame_count)))
                torch.save({
                    'replay_mem' : replay_mem,
                    'end_eps' : curr_eps,
                    'end_episode' : episode_num,
                    'curr_frame_count' : curr_frame_count,
                    'optim' : optimizer.state_dict()
                }, os.path.join(states_dir, str(curr_frame_count)))

        #
        #Other info
        #

        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        food_per_frame = episode_food_count / episode_frame_count if episode_frame_count > 0 else 0

        print()
        print('Average loss:', avg_loss)
        print('Replay memory size:', len(replay_mem))
        print('Eps:', curr_eps)
        print('Average speed:', str(avg_speed) + 'it/s')
        print('Episode food count:', episode_food_count)
        print('Episode frame count:', episode_frame_count)

        # Log to TensorBoard
        writer.add_scalar('Training/Loss', avg_loss, curr_frame_count)
        writer.add_scalar('Training/Epsilon', curr_eps, curr_frame_count)
        writer.add_scalar('Training/ReplayMemorySize', len(replay_mem), curr_frame_count)
        writer.add_scalar('Episode/FoodCount', episode_food_count, episode_num)
        writer.add_scalar('Episode/FrameCount', episode_frame_count, episode_num)
        writer.add_scalar('Episode/FoodPerFrame', food_per_frame, episode_num)
        if avg_speed is not None:
            writer.add_scalar('Performance/Speed_it_per_s', avg_speed, curr_frame_count)

        # Update final metrics for hparams summary
        final_metrics['final_food_count'] = episode_food_count
        final_metrics['final_frame_count'] = episode_frame_count
        final_metrics['final_food_per_frame'] = food_per_frame
        final_metrics['final_loss'] = avg_loss

        #
        #Episode loop routine
        #

        episode_num += 1

    # Log final hyperparameters and metrics summary to TensorBoard
    # This creates a nice comparison table in the HPARAMS tab
    writer.add_hparams(
        hparams,
        final_metrics
    )

    # Close TensorBoard writer
    writer.close()

    print()
    print('Done!')
    print(f'Final metrics: food={final_metrics["final_food_count"]}, '
          f'frames={final_metrics["final_frame_count"]}, '
          f'efficiency={final_metrics["final_food_per_frame"]:.4f}')