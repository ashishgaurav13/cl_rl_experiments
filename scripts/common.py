import numpy as np

def read_eval_lines(fname, clip = [-np.inf, np.inf]):
    assert(clip[0] < clip[1])
    f = open(fname, "r")
    lines = [item.strip() for item in f.readlines() if item.strip != ""]
    lines = [list(map(lambda x: np.clip(x, clip[0], clip[1]), eval(item))) \
        for item in lines if item.startswith("[") and item.endswith("]")]
    f.close()
    return lines

def smoothen(x, n, func = np.average):
    assert(len(x) >= n)
    ret = []
    i = 0
    while i+n <= len(x):
        ret += [func(x[i:i+n])]
        i += 1
    assert(len(ret) == len(x)-n+1)
    return ret

def combine_streams(streams, func = np.average, strip_unequal = True):
    assert(type(streams) == list)
    out_stream = []
    for stream in streams:
        for eid, element in enumerate(stream):
            if eid >= len(out_stream):
                out_stream.append([])
                assert(eid < len(out_stream))
            out_stream[eid].append(element)
    len_first = len(out_stream[0])
    out_stream2 = []
    for cstream in out_stream:
        if len(cstream) != len_first and strip_unequal:
            continue
        out_stream2.append(func(cstream))
    return np.array(out_stream2)