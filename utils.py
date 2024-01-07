import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(x, scores, eps_history, filename):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, eps_history)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Epsilon')
    ax1.set_title('Epsilon Decay')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, scores)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score')
    ax2.set_title('Score per Episode')

    plt.savefig(filename)
    plt.show()
