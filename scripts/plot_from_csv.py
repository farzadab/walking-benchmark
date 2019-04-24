import numpy as np
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
    parser.add_argument('--name_regex', type=str, default='')
    args = parser.parse_args()

    N = len(args.columns)
    nrows = math.floor(math.sqrt(N))
    plot = Plot(nrows=nrows, ncols=math.ceil(N / nrows))
    plots = []
    for column in args.columns:
        plots.append(LinePlot(parent=plot, ylabel=column, xlabel='TotalSteps', num_scatters=len(args.load_paths)))
    
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
        for j, column in enumerate(args.columns):
            with open(os.path.join(path, 'progress.csv'), 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                points = np.array([(int(row['TotalStep']), float(row[column])) for row in reader])
                plots[j].update(points, line_num=i)
    
    plt.show()


if __name__ == '__main__':
    main()
