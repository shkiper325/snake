#!/bin/bash

# Start TensorBoard to visualize training and evaluation metrics
# This will monitor the ./runs directory for TensorBoard logs

echo "Starting TensorBoard..."
echo "Open your browser at http://localhost:6006"
echo "Press Ctrl+C to stop TensorBoard"
echo ""

tensorboard --logdir=./runs --port=6006
