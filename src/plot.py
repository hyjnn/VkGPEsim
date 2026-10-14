#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load("results/test.npy")
    xs = np.arange(data.size) - data.size // 2

    fig, ax = plt.subplots(dpi = 300)

    ps = np.pow(np.abs(data), 2)
    ax.plot(xs, ps)

    mean = np.average(xs, weights = ps)
    print("Mean = {}, standard deviation = {}".format(mean, np.sqrt(np.average((xs - mean)**2, weights = ps))))

    plt.show()
