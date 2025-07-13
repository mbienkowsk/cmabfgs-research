import glob

import numpy as np


def load_results_from_csv(path: str):
    """Given a csv of x,y pairs, load it and return two arrays"""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def load_results_from_directory(dir_path: str):
    """Given a directory, load all CSVs and return lists of x and y arrays."""
    xx = []
    yy = []
    for csv_file in glob.glob(f"{dir_path}/*.csv"):
        x, y = load_results_from_csv(csv_file)
        xx.append(x)
        yy.append(y)
    return xx, yy


def load_and_interpolate_results(dir_path: str):
    """Given a directory with csv x/y pairs, load all of them and return the interpolated values"""
    xx, yy = load_results_from_directory(dir_path)
    xmax = max(x[-1] for x in xx)
    return average_interpolated_values(yy, xx, xmax)


def average_interpolated_values(values, evals, maxevals):
    """Interpolates values to the same length and averages them.
    Returns both the x values and the y values to later plot."""

    shortest = min(len(v) for v in values)

    x = np.linspace(0, maxevals, shortest)

    return x, np.mean(
        np.array([np.interp(x, e, v) for v, e in zip(values, evals)]), axis=0
    )
