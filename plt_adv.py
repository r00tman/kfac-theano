#!/usr/bin/python3
import sys

import matplotlib.pyplot as plt


def safe_to_float(x):
    try:
        return float(x)
    except ValueError:
        return x


def load_csv(fn):
    with open(fn) as f:
        header = f.readline().strip().split(',')

        data = [[safe_to_float(cell) for cell in line.strip().split(',')]
                for line in f.readlines()]
        data_by_cols = {col_name: [row[col] for row in data]
                        for col, col_name in enumerate(header)}

        return data_by_cols


plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.dpi"] = 110
data = [(fn, load_csv(fn)) for fn in sys.argv[2:]]
for col in sys.argv[1].split(','):
    for fn, vals in data:
        if col in vals:
            plt.plot(vals[col], label='%s - %s' % (fn, col))
        else:
            print("'%s' is not in '%s', ignoring" % (col, fn))

plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()
