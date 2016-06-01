#!/usr/bin/python3
import os
from itertools import accumulate
import matplotlib.pyplot as plt
import sys

def load_trace(path_to_file):
    properties = []
    with open(path_to_file, 'r') as trace_file:
        for row in trace_file:
            elements = map(float, row.split())
            properties.append(elements)

    tss = [list(a) for a in zip(*properties)]

    tss[1] = list(accumulate(tss[1]))

    # We don't care about the iterations.
    return tss[1:]


if len(sys.argv) < 2:
    print("./plot.py path_to_stereo_folder")


path_to_folder = sys.argv[1]

path_to_qp_trace = os.path.join(path_to_folder, "out_tracing-qp.txt")
path_to_mf_trace = os.path.join(path_to_folder, "out_tracing-mf.txt")
path_to_qpcccp_cv_trace = os.path.join(path_to_folder, "out_tracing-proper_qpcccp_cv.txt")
path_to_qpcccp_trace = os.path.join(path_to_folder, "out_tracing-qpcccp.txt")
path_to_lp_trace = os.path.join(path_to_folder, "out_tracing-sg_lp.txt")


qp_trace = load_trace(path_to_qp_trace)
mf_trace = load_trace(path_to_mf_trace)
qpcccp_cv_trace = load_trace(path_to_qpcccp_cv_trace)
qpcccp_trace = load_trace(path_to_qpcccp_trace)
lp_trace = load_trace(path_to_lp_trace)

plt.figure(1)
plt.title("Assignment Energy as a function of time")
plt.plot(qp_trace[0], qp_trace[1], 'ro', label="QP")
plt.plot(mf_trace[0], mf_trace[1], 'bo', label="MF")
plt.plot(qpcccp_cv_trace[0], qpcccp_cv_trace[1], 'go', label="CCV")
plt.plot(qpcccp_trace[0], qpcccp_trace[1], 'mo', label="CCCP")
plt.plot(lp_trace[0], lp_trace[1], 'yo', label="lp")
plt.legend()


plt.show()
