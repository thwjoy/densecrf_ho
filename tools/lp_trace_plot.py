#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import pandas
import sys


def generate_graphs(path_to_folder):
    path_to_data = os.path.join(path_to_folder, 'trace.txt')
    column_names = ['fractional', 'max_rounded', 'int_rounded']
    df = pandas.read_csv(path_to_data, sep='\t', index_col=0,
                         header=None, names=column_names)

    fig = plt.figure()
    fractional = fig.add_subplot(211)
    rounded = fig.add_subplot(212)
    fractional.plot(df.index, df.fractional, label='Fractional Energy')
    fractional.legend()
    rounded.plot(df.index, df.max_rounded, label='Argmax')
    rounded.plot(df.index, df.int_rounded, label='Random')
    rounded.legend()

    path_to_output = os.path.join(path_to_folder, 'graphs.png')
    fig.savefig(path_to_output)


if __name__ == "__main__":
    root_folder = sys.argv[1]
    all_subdir = os.listdir(root_folder)
    for sub_dir in all_subdir:
        folder_path = os.path.join(root_folder, sub_dir)
        if os.path.exists(os.path.join(folder_path, 'trace.txt')):
            generate_graphs(folder_path)
