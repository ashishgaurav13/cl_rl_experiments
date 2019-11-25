import matplotlib.pyplot as plt
import argparse
import scripts
import numpy as np
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-f", type = str, required = True)
parser.add_argument("-clip", default = False, action = "store_true")
parser.add_argument("-save", default = "plot.png", type = str)
parser.add_argument("-show", default = False, action = "store_true")
args = parser.parse_args()

smoothen = 25
args.f = glob.glob(args.f)
all_lines = []
clip = [-np.inf, np.inf] if not args.clip else [-100., 100.]

for f in args.f:
    lines = scripts.common.read_eval_lines(f, clip = clip)
    lines = list(map(lambda x: round(np.mean(x), 2), lines))
    # average envs 0-3 eval scores and smoothen last few epochs
    lines = scripts.common.smoothen(lines, smoothen, np.average)
    all_lines += [lines]

# average across seeds
average_stream = scripts.common.combine_streams(all_lines, np.average)
max_stream = scripts.common.combine_streams(all_lines, np.max)
# max_stream = average_stream + np.std(max_stream-average_stream)
min_stream = scripts.common.combine_streams(all_lines, np.min)
# min_stream = average_stream - np.std(average_stream-min_stream)
x = list(range(len(average_stream)))
plt.plot(x, average_stream, color = 'blue')
plt.fill_between(x, min_stream, max_stream, facecolor = 'blue', alpha = 0.2)
if args.show:
    plt.show()
else:
    plt.savefig(args.save)
    print("Saved to %s" % args.save)