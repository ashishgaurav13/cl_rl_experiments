import sys, datetime, os
import json, collections
import time

# https://stackoverflow.com/questions/17866724/
# python-logging-print-statements-while-having-them-print-to-stdout
# Print something while also logging to another file.
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
        self.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Use __main__.__file__, timestamp, seed to create a .txt
# log file, such that when something is printed, it shows up
# on stdout as well in that .txt log file; If overwrite_name
# is given, then we replace __main__.__file__ with that
def log(logfile, seed = 0):
    if not os.path.exists("logs"):
        os.mkdir("logs")
        print("os.mkdir: logs")
    logfile = logfile + datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S") + "_seed%d.txt" % seed
    logfile = open(os.path.join("logs", logfile), "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)

# Gives an initial time using time.time()
def timer():
    return time.time()

# Prints elapsed time in human-readable units, given a start
# datetime object
def timer_done(start_time):
    return str(datetime.timedelta(seconds = int(time.time()-start_time)))

def nowarnings():
    import warnings
    warnings.filterwarnings("ignore")

def dict_sort(x):
    if type(x) != dict:
        return x
    def convert_numeric(x):
        try:
            ret = float(x)
        except:
            return x
        else:
            return ret
    return dict(collections.OrderedDict(
        sorted([(item[0], dict_sort(item[1])) for item in x.items()],
            key = lambda x: convert_numeric(x[0]))))

def json_dump(x, f = "log.txt", show = False):
    assert(type(x) == dict)
    x = dict_sort(x)
    ff = open(f, "w")
    json.dump(x, ff, indent = 2)
    ff.close()
    if show:
        print(json.dumps(x, indent = 2))

def json_load(f = "log.txt", show = False):
    if not os.path.exists(f):
        json_dump({}, f)
    ff = open(f, "r")
    ret = json.load(ff)
    ff.close()
    ret = dict_sort(ret)
    json_dump(ret, f)
    if show:
        print(json.dumps(ret, indent = 2))
    return ret