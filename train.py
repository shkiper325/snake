import time
import random
import os
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

def format_time(seconds):
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def print_header():
    """Print training header."""
    print()
    print("=" * 70)
    print("  SNAKE DQN TRAINING")
    print("=" * 70)

def print_config(device, models_dir, states_dir, logs_dir, hparams, env):
    """Print configuration summary."""
    print()
    print(f"  Device:     {device}")
    print(f"  Models:     {models_dir}")
    print(f"  States:     {states_dir}")
    print(f"  Logs:       {logs_dir}")
    print()
    print("-" * 70)
    print("  HYPERPARAMETERS")
    print("-" * 70)
    print(f"  gamma={hparams['gamma']:<8} batch_size={hparams['batch_size']:<6} lr={hparams['learning_rate']}")
    print(f"  eps: {hparams['eps_start']:.2f} -> {hparams['eps_end']:.2f} (decay={hparams['eps_decay_time']})")
    print(f"  frames={hparams['frame_count']:,}  episode_depth={hparams['episode_depth']:,}")
    print(f"  Q_targ update every {hparams['Q_targ_update_freq']:,} frames")
    print()
    print("-" * 70)
    print("  ENVIRONMENT")
    print("-" * 70)
    print(f"  Field: {env.field_size[0]}x{env.field_size[1]}  Snake: {env.snake_init_len}  Food: {env.food_count}")
    print(f"  Rewards: food={env.food_score} death={env.death_score} survive={env.survive_score}")
    print("=" * 70)
    print()

DEVICE = "CUDA" if USE_CUDA else "CPU"

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
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory for models and states (default: current directory)')
    args = parser.parse_args()

    # Set headless mode via environment variable if specified
    if args.headless:
        os.environ['SNAKE_HEADLESS'] = '1'

    #Save/load dirs
    if args.out_dir:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        models_dir = os.path.join(args.out_dir, 'models')
        states_dir = os.path.join(args.out_dir, 'states')
    else:
        models_dir = './models'
        states_dir = './states'

    logs_dir = './runs'

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

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
    cold_start = os.listdir(models_dir) == []
    if cold_start:
        Q.apply(utils.init_weights)
        Q_targ.load_state_dict(Q.state_dict())
    else:
        weights_file_path, state_file_path = utils.find_prev_state_files(models_dir, states_dir)
        weights = torch.load(weights_file_path, weights_only=True)
        prev_state = torch.load(state_file_path, weights_only=False)
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

    # Define all hyperparameters dict
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

    # Print beautiful header and config
    print_header()
    mode_str = "HEADLESS" if args.headless else "DISPLAY"
    start_str = "COLD START" if cold_start else f"RESUMING from episode {prev_state['end_episode']}"
    print(f"  Mode: {mode_str} | {start_str}")
    print_config(DEVICE, models_dir, states_dir, logs_dir, hparams, env)

    # Log hyperparameters to TensorBoard

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

    # Log output directories
    dirs_text = f'''
Output Directories:
- Models: {models_dir}
- States: {states_dir}
- Logs: {logs_dir}
'''
    writer.add_text('Directories/Paths', dirs_text, 0)

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
    training_start_time = time.time()

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

    # Track best performance
    best_food = 0

    print("Starting training loop...")
    print()

    while True:
        if curr_frame_count > frame_count:
            break

        losses = []
        episode_food_count = 0
        episode_frame_count = 0

        env.new_game()

        last_frames = [np.full((10, 10, 3), fill_value=255, dtype=np.uint8) for _ in range(k - 1)]
        last_frames.append(env.screenshot())

        curr_state = prepare_state(last_frames)

        for t in range(episode_depth):
            #
            # Acting
            #

            if env.finished():
                break

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
                Q_targ.load_state_dict(Q.state_dict())

            #
            #Saving if needed
            #

            if curr_frame_count % save_freq == 0:
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
        # Episode summary
        #

        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        food_per_frame = episode_food_count / episode_frame_count if episode_frame_count > 0 else 0

        # Track best performance
        if episode_food_count > best_food:
            best_food = episode_food_count

        # Calculate progress and ETA
        progress = curr_frame_count / frame_count * 100
        elapsed = time.time() - training_start_time
        if curr_frame_count > 0:
            eta = elapsed / curr_frame_count * (frame_count - curr_frame_count)
        else:
            eta = 0

        # Build progress bar
        bar_width = 20
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Speed string
        speed_str = f"{avg_speed:.1f} it/s" if avg_speed else "..."

        # Print compact episode summary
        print(f"[{bar}] {progress:5.1f}% | "
              f"Ep {episode_num:4d} | "
              f"Food: {episode_food_count:2d} (best:{best_food:2d}) | "
              f"Steps: {episode_frame_count:4d} | "
              f"Loss: {avg_loss:.4f} | "
              f"ε: {curr_eps:.3f} | "
              f"{speed_str} | "
              f"ETA: {format_time(eta)}")

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

        episode_num += 1

    # Log final hyperparameters and metrics summary to TensorBoard
    writer.add_hparams(hparams, final_metrics)
    writer.close()

    # Print final summary
    total_time = time.time() - training_start_time
    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time:     {format_time(total_time)}")
    print(f"  Episodes:       {episode_num}")
    print(f"  Frames:         {curr_frame_count:,}")
    print(f"  Best food:      {best_food}")
    print(f"  Final loss:     {final_metrics['final_loss']:.4f}")
    print(f"  Final epsilon:  {curr_eps:.4f}")
    print()
    print(f"  Models saved to: {models_dir}")
    print(f"  TensorBoard:     tensorboard --logdir={logs_dir}")
    print("=" * 70)