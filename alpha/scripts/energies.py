#!/usr/bin/env python
import sys
import os
import glob
from collections import namedtuple

Props = namedtuple("Props", ['image_id', 'timing', 'frac_energy', 'int_energy' ])

def get_properties_per_images(folder_path):
    method = folder_path.split('/')[-2]
    tracing = False
#    if method.find('tracing') >= 0:
#        tracing = True
    all_txts = glob.glob(folder_path + '*.txt')
    props = []
    for txt_file in all_txts:
        with open(txt_file, 'r') as txt:
            image_id = os.path.basename(txt_file)[:-4]
            if tracing:
                content = txt.read().strip().split('\n')
                last_line = content[-1].split('\t')
                timing = float(last_line[1])
                frac_energy = float(last_line[2])   # also int-energy
                integer_energy = float(last_line[2])
            else:
                content = txt.read().strip().split('\t')
                timing = float(content[0])
                frac_energy = float(content[1])
                integer_energy = float(content[2])

            props.append(Props(image_id, timing, frac_energy, integer_energy))
    props.sort()

    return props





def main():
    if len(sys.argv) < 3:
        print "Missing argument:"
        print "energies.py /path/to/results/folder/ /path/to/other/results/folder"
        return 1

    path_to_first_folder = sys.argv[1]
    path_to_second_folder = sys.argv[2]

    first = get_properties_per_images(path_to_first_folder)
    second = get_properties_per_images(path_to_second_folder)

    frac_better_1 = 0
    frac_better_2 = 0
    frac_better_same = 0

    int_better_1 = 0
    int_better_2 = 0
    int_better_same = 0

    faster_1 = 0
    faster_2 = 0
    faster_same = 0

    total_timing_1 = total_timing_2 = 0
    total_frac_energy_1 = total_frac_energy_2 = 0
    total_int_energy_1 = total_int_energy_2 = 0
    for prop1, prop2 in zip(first, second):
        #print(str(prop1) + " # " + str(prop2))
        if prop1.timing < prop2.timing:
            faster_1 += 1
        elif prop1.timing == prop2.timing:
            faster_same += 1
        else:
            faster_2 += 1

        if prop1.frac_energy < prop2.frac_energy:
            frac_better_1 += 1
        elif prop1.frac_energy == prop2.frac_energy:
            frac_better_same += 1
        else:
            frac_better_2 += 1

        if prop1.int_energy < prop2.int_energy:
            int_better_1 += 1
        elif prop1.int_energy == prop2.int_energy:
            int_better_same += 1
        else:
            int_better_2 += 1

        total_timing_1 += prop1.timing
        total_timing_2 += prop2.timing
        total_frac_energy_1 += prop1.frac_energy
        total_frac_energy_2 += prop2.frac_energy
        total_int_energy_1 += prop1.int_energy
        total_int_energy_2 += prop2.int_energy

    print "First method is faster in %s%% and similar in %s%% of the cases" % ((faster_1 *100) / float(faster_1 + faster_2 + faster_same), (faster_same*100) / float(faster_1 + faster_2 + faster_same))
    print "First method reach lower fractional energy in %s%% and same in %s%% of the cases" % ((frac_better_1 *100) / float(frac_better_1 + frac_better_2 + frac_better_same), (frac_better_same *100) / float(frac_better_1 + frac_better_2 + frac_better_same))
    print "First method reach lower integer energy in %s%% and same in %s%% of the cases" % ((int_better_1 *100) / float(int_better_1 + int_better_2 + int_better_same), (int_better_same *100) / float(int_better_1 + int_better_2 + int_better_same))

    print "Average timings: %s vs %s" % (total_timing_1/float(len(first)), total_timing_2/float(len(second)))
    print "Average fractional energy: %s vs %s" % (total_frac_energy_1/float(len(first)), total_frac_energy_2/float(len(second)))
    print "Average Integer energy: %s vs %s" % (total_int_energy_1/float(len(first)), total_int_energy_2/float(len(second)))

    return 0

if __name__ == "__main__":
    sys.exit(main())
