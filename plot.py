import json
import os

import matplotlib.pyplot as plt
import matplotlib

# PNG plotting disabled - use TensorBoard instead
# Run `./start_tensorboard.sh` to visualize all metrics

if __name__ == '__main__':
    print("PNG plotting is disabled.")
    print("Use TensorBoard for visualization:")
    print("  1. Run: ./start_tensorboard.sh")
    print("  2. Open: http://localhost:6006")
    print("")
    print("All metrics are available in TensorBoard after running train.py or stats.py")

    # fd = open(os.path.join('result.json'), 'r')
    # json_content = json.loads(fd.read())
    # fd.close()

    # foods = json_content['mean_food']
    # tsses = json_content['states']

    # plt.clf()
    # matplotlib.rcParams.update({'font.size': 22})
    # plt.figure(figsize=(25, 20), dpi=80)
    # plt.plot(tsses, foods)
    # plt.savefig('scores.png')
    