import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import argparse
import math
import csv
import re
import os

from utils.plots import Plot, LinePlot, plt


def main():
    plt.ioff()

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_paths', type=str, nargs='+')
    parser.add_argument('--columns', type=str, nargs='+')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--smoothing', type=float, default=1.2)
    parser.add_argument('--name_regex', type=str, default='')
    args = parser.parse_args()

    N = len(args.columns)
    nrows = math.floor(math.sqrt(N))
    plot = Plot(nrows=nrows, ncols=math.ceil(N / nrows), title='Results with smoothing %.1f' % args.smoothing)
    plots = []
    for column in args.columns:
        plots.append(
            LinePlot(
                parent=plot,
                ylabel=column, xlabel='TotalSteps',
                alpha=args.alpha,
                num_scatters=len(args.load_paths),
            )
        )
    
    if args.name_regex:
        legends = [re.findall(args.name_regex, path)[0] for path in args.load_paths]
        legend_paths = sorted(zip(legends, args.load_paths))
        legends = [x[0] for x in legend_paths]
        args.load_paths = [x[1] for x in legend_paths]        
    else:
        common_prefix = os.path.commonprefix(args.load_paths)
        print('Ignoring the prefix (%s) in the legend' % common_prefix)
        legends = [path[len(common_prefix):] for path in args.load_paths]

    plots[0].subplot.legend(legends)
    
    for i, path in enumerate(args.load_paths):
        print('Loading ... ', path)
        df = pd.read_csv(os.path.join(path, 'progress.csv'))
        for j, column in enumerate(args.columns):
            df[column] = gaussian_filter1d(df[column], sigma=args.smoothing)
            plots[j].update(df[['TotalStep', column]].values, line_num=i)
    
    plt.show()


if __name__ == '__main__':
    main()
