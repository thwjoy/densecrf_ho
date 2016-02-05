#!/usr/bin/env python
import sys
import os
import glob
from collections import namedtuple

Props = namedtuple("Props", ['image_id', 'timing', 'energy'])

def get_properties_per_images(folder_path):
    all_txts = glob.glob(folder_path + '*.txt')
    props = []
    for txt_file in all_txts:
        with open(txt_file, 'r') as txt:
            image_id = os.path.basename(txt_file)[:-4]
            content = txt.read().strip().split('\t');
            timing = float(content[0])
            final_energy = float(content[1])
            props.append(Props(image_id, timing, final_energy))
    props.sort()

    return props





def main():
    if len(sys.argv) < 2:
        print "Missing argument:"
        print "energies.py /path/to/results/folder/ /path/to/other/results/folder"
        return 1

    path_to_first_folder = sys.argv[1]
    path_to_second_folder = sys.argv[2]

    first = get_properties_per_images(path_to_first_folder)
    second = get_properties_per_images(path_to_second_folder)

    better_1 = 0
    better_2 = 0
    better_same = 0

    faster_1 = 0
    faster_2 = 0
    faster_same = 0

    total_timing_1 = total_timing_2 = 0
    total_energy_1 = total_energy_2 = 0
    for prop1, prop2 in zip(first, second):
        if prop1.timing < prop2.timing:
            faster_1 += 1
        elif prop1.timing == prop2.timing:
            faster_same += 1
        else:
            faster_2 += 1

        if prop1.energy < prop2.energy:
            better_1 += 1
        elif prop1.energy == prop2.energy:
            better_same += 1
        else:
            better_2 += 1

        total_timing_1 += prop1.timing
        total_timing_2 += prop2.timing
        total_energy_1 += prop1.energy
        total_energy_2 += prop2.energy

    print "First method is faster in %s%% and similar in %s%% of the cases" % ((faster_1 *100) / (faster_1 + faster_2 + faster_same), (faster_same*100) / (faster_1 + faster_2 + faster_same))
    print "First method reach lower energy in %s%% and same in %s%% of the cases" % ((better_1 *100) / (better_1 + better_2 + better_same), (better_same *100) / (better_1 + better_2 + better_same))

    print "Average timings: %s vs %s" % (total_timing_1/float(len(first)), total_timing_2/float(len(second)))
    print "Average energy: %s vs %s" % (total_energy_1/float(len(first)), total_energy_2/float(len(second)))

    return 0

if __name__ == "__main__":
    sys.exit(main())
