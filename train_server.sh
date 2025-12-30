#!/bin/bash

# Example script for training on a headless server
# This script runs training in headless mode with nohup to continue after SSH disconnect

echo "Starting headless training on server..."
echo "Logs will be saved to train.log"
echo ""

# Run training in background with headless mode
nohup python train.py --headless \
    --frame-count 7008000 \
    --batch-size 32 \
    --eps-decay-time 0.5 \
    > train.log 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "To monitor progress: tail -f train.log"
echo "To stop training: kill $PID"
echo ""
echo "TensorBoard logs are in ./runs/"
echo "To view TensorBoard, run on your local machine:"
echo "  ssh -L 6006:localhost:6006 user@server"
echo "  then run: ./start_tensorboard.sh"
