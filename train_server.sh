#!/bin/bash

# Example script for training on a headless server
# This script runs training in headless mode with nohup to continue after SSH disconnect

# Set experiment name (default: experiment1)
EXPERIMENT_NAME="${1:-experiment1}"

echo "Starting headless training on server..."
echo "Experiment: $EXPERIMENT_NAME"
echo "Logs will be saved to ${EXPERIMENT_NAME}_train.log"
echo ""

# Run training in background with headless mode
nohup python train.py --headless \
    --out-dir "$EXPERIMENT_NAME" \
    --frame-count 7008000 \
    --batch-size 32 \
    --eps-decay-time 0.5 \
    > "${EXPERIMENT_NAME}_train.log" 2>&1 &

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
