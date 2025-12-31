#!/bin/bash

# Example script for training on a headless server
# This script runs training in headless mode with nohup to continue after SSH disconnect

echo "Starting headless training on server..."
echo ""

# Run training in background with headless mode
SNAKE_HEADLESS=1 nohup python train.py --headless \
    --out-dir "div_1" \
    --frame-count 7008000 \
    --batch-size 32 \
    --eps-decay-time 0.5 \
    --q-targ-update-freq 32000 \
    > div_1_train.log 2>&1 &

SNAKE_HEADLESS=1 nohup python train.py --headless \
    --out-dir "div_10" \
    --frame-count 700800 \
    --batch-size 32 \
    --eps-decay-time 0.5 \
    --q-targ-update-freq 3200 \
    > div_10_train.log 2>&1 &

SNAKE_HEADLESS=1 nohup python train.py --headless \
    --out-dir "div_5" \
    --frame-count 1401600 \
    --batch-size 32 \
    --eps-decay-time 0.5 \
    --q-targ-update-freq 6400 \
    > div_5_train.log 2>&1 &

SNAKE_HEADLESS=1 nohup python train.py --headless \
    --out-dir "div_4" \
    --frame-count 1752000 \
    --batch-size 32 \
    --eps-decay-time 0.5 \
    --q-targ-update-freq 8000 \
    > div_4_train.log 2>&1 &

SNAKE_HEADLESS=1 nohup python train.py --headless \
    --out-dir "div_2" \
    --frame-count 3504000 \
    --batch-size 32 \
    --eps-decay-time 0.5 \
    --q-targ-update-freq 16000 \
    > div_2_train.log 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "To monitor progress: tail -f ${EXPERIMENT_NAME}_train.log"
echo "To stop training: kill $PID"
echo ""
echo "Output directories:"
echo "  Models: ./$EXPERIMENT_NAME/models/"
echo "  States: ./$EXPERIMENT_NAME/states/"
echo "  TensorBoard logs: ./runs/"
echo ""
echo "To view TensorBoard, run on your local machine:"
echo "  ssh -L 6006:localhost:6006 user@server"
echo "  then run: ./start_tensorboard.sh"
echo ""
echo "Usage: $0 [experiment_name]"
echo "Example: $0 large_batch    # Creates ./large_batch/ directory"
