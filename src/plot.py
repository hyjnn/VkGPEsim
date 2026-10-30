#!/bin/python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x: np.ndarray, A: float, mu: float, std: float) -> np.ndarray:
    return A * np.exp(-.5 * ((x - mu) / std)**2)

def parse_terminal_args() -> tuple[str, bool]:
    parser = argparse.ArgumentParser(prog = "GPEplot", description = "Simple plotter for gpe simulation results")
    parser.add_argument('-f', '--filename', default = "results/test.npy")
    parser.add_argument('--nofit', action = 'store_true')

    args = parser.parse_args()

    return args.filename, args.nofit

if __name__ == "__main__":
    filename, no_fit = parse_terminal_args()

    data = np.load(filename)
    # xs = np.arange(data.size) - data.size // 2
    dx = 10 / 800
    xmin = -1 * data.size // 2 * dx
    xmax = (data.size // 2 - 1) * dx
    xs = np.linspace(xmin, xmax, data.size)

    fig, ax = plt.subplots(dpi = 300)

    ps = np.pow(np.abs(data), 2)
    ax.plot(xs, ps)

    mean = np.average(xs, weights = ps)
    print("Mean = {}, standard deviation = {}".format(mean, np.sqrt(np.average((xs - mean)**2, weights = ps))))

    if not no_fit:
        par, cov = curve_fit(gaussian, xs, ps)
        ax.plot(xs, gaussian(xs, *par))
        print("Fit mean = {}, fit standard deviation = {}".format(par[1], par[2]))

    plt.show()
