#!/usr/bin/env python3
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



def main():
    if len(sys.argv) < 2 or "--help" in sys.argv:
        print("./plot.py path_to_stereo_folder")
        return 0


    path_to_folder = sys.argv[1]

    path_to_qp_trace = os.path.join(path_to_folder, "out_tracing-qp.txt")
    path_to_mf_trace = os.path.join(path_to_folder, "out_tracing-mf.txt")
    path_to_qpcccp_cv_trace = os.path.join(path_to_folder, "out_tracing-proper_qpcccp_cv.txt")
    path_to_qpcccp_trace = os.path.join(path_to_folder, "out_tracing-qpcccp.txt")
    path_to_lp_trace = os.path.join(path_to_folder, "out_tracing-sg_lp.txt")
    path_to_ccv_trace = os.path.join(path_to_folder, "out_tracing-only-ccv.txt")


    qp_trace = load_trace(path_to_qp_trace)
    mf_trace = load_trace(path_to_mf_trace)
    qpcccp_cv_trace = load_trace(path_to_qpcccp_cv_trace)
    qpcccp_trace = load_trace(path_to_qpcccp_trace)
    lp_trace = load_trace(path_to_lp_trace)
    ccv_trace = load_trace(path_to_ccv_trace)

    plt.rc('text', usetex=True)
    ax = plt.figure(1)
    plt.title("Assignment Energy as a function of time")
    plt.semilogx(mf_trace[0], mf_trace[1], 'b-', label="MF")
    plt.semilogx(qp_trace[0], qp_trace[1], 'r-', label="QP")
    plt.semilogx(ccv_trace[0], ccv_trace[1], 'c-', label="DC_{neg}")
    plt.semilogx(qpcccp_cv_trace[0], qpcccp_cv_trace[1], 'g-', label="QP-DC_{neg}")
    plt.semilogx(qpcccp_trace[0], qpcccp_trace[1], 'm-', label="QP-CCCP")
    plt.semilogx(lp_trace[0], lp_trace[1], 'y-', label="QP-DC_{neg}-LP")
    plt.xlim([0.1, 300])
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
