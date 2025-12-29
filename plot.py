import json
import os

import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    fd = open(os.path.join('result.json'), 'r')
    json_content = json.loads(fd.read())
    fd.close()

    foods = json_content['mean_food']
    tsses = json_content['states']

    plt.clf()
    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(25, 20), dpi=80)
    plt.plot(tsses, foods)
    plt.savefig('scores.png')
    