#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load("results/test.npy")
    xs = np.linspace(0, 1, data.size)

    fig, ax = plt.subplots(dpi = 300)

    ax.plot(xs, np.abs(data))

    plt.show()
