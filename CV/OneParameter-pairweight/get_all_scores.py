import matlab.engine
import os


def evaluate_segmentation(pairwise_weight):
    eng = matlab.engine.start_matlab()
    path_to_results = "/data/CV/OneParameter-pairweight/" + \
        str(pairwise_weight) + "/mf25"
    ret = eng.voc_test(path_to_results)

    # This returns the value of the average accuracy. Spearmint
    # minimize things and we want to maximise this, so we should
    # return the negative
    return - ret


top_directory = "/data/CV/OneParameter-pairweight/"
all_pairwise_weight = os.listdir(top_directory)

all_results = []
for pairwise_weight in all_pairwise_weight:
    all_results.append(
        (evaluate_segmentation(pairwise_weight), float(pairwise_weight)))

print sorted(all_results)
