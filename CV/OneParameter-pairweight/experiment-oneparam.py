import subprocess
import matlab.engine


def generate_segmentation(pairwise_weight):
    path_to_executable = '/home/rudy/workspace/densecrf/build/alpha/cv-script'
    split = "Validation"
    dataset = "Pascal2010"
    weight_passed = str(pairwise_weight)
    results_path = "/data/CV/OneParameter-pairweight/" + weight_passed + "/"
    subprocess.call([path_to_executable,
                     split,
                     dataset,
                     results_path,
                     weight_passed])



def evaluate_segmentation(pairwise_weight):
    eng = matlab.engine.start_matlab()
    path_to_results = "/data/CV/OneParameter-pairweight/" + str(pairwise_weight) + "/mf5"
    ret = eng.voc_test(path_to_results)

    # This returns the value of the average accuracy. Spearmint
    # minimize things and we want to maximise this, so we should
    # return the negative
    return - ret


# Write a function like this called 'main'
def main(job_id, params):
    pairwise_weight = params['pairwise_weight'][0]
    generate_segmentation(pairwise_weight)
    return evaluate_segmentation(pairwise_weight)
