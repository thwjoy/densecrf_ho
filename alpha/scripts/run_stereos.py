#!/usr/bin/python3
import os
import sh
import sys

trace_to_collect = ["tracing-mf",
                    "tracing-qp",
                    "tracing-qpcccp",
                    "tracing-proper_qpcccp_cv",
                    "tracing-sg_lp",
                    "tracing-only-ccv"]


def main():
    if len(sys.argv) < 3 or "--help" in sys.argv:
        print("./run_stereos.py stereo_binary path_to_stereo_folder")
        return
    path_to_stereo_binary = sys.argv[1]
    stereo_command = sh.Command(path_to_stereo_binary)

    path_to_stereo_folder = sys.argv[2]
    stereo_experiments = os.listdir(path_to_stereo_folder)
    stereo_experiments = [os.path.join(path_to_stereo_folder,
                                       exp) for exp in stereo_experiments]

    for trace in trace_to_collect:
        for exp in stereo_experiments:
            stereo_command(exp, trace)

if __name__ == "__main__":
    main()
