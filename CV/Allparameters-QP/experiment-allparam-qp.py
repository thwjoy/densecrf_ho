import subprocess
import matlab.engine
import os

def generate_segmentation(spc_std, spc_potts,
                          bil_spcstd, bil_colstd, bil_potts):
    path_to_executable = '/data/Experiments/experiment-allparam-qp/cv-script'
    split = "Validation"
    dataset = "Pascal2010"

    exp_path = '/'.join(map(str, [spc_std, spc_potts, bil_spcstd,
                                  bil_colstd, bil_potts]))

    results_path = "/data/CV/Allparams/" + exp_path + "/"
    try:
        os.makedirs(results_path)
    except OSError:
        pass
    subprocess.call([path_to_executable,
                     split,
                     dataset,
                     results_path,
                     str(spc_std), str(spc_potts),
                     str(bil_spcstd), str(bil_colstd), str(bil_potts)])


def evaluate_segmentation(spc_std, spc_potts,
                          bil_spcstd, bil_colstd, bil_potts):
    eng = matlab.engine.start_matlab()
    exp_path = '/'.join(map(str, [spc_std, spc_potts, bil_spcstd,
                                  bil_colstd, bil_potts]))
    path_to_results = "/data/CV/Allparams/" + exp_path + "/lrqp"
    ret = eng.voc_test(path_to_results)

    # This returns the value of the average accuracy. Spearmint
    # minimize things and we want to maximise this, so we should
    # return the negative
    return - ret


# Write a function like this called 'main'
def main(job_id, params):
    spc_std = params['spc_std'][0]
    spc_potts = params['spc_potts'][0]
    bil_spcstd = params['bil_spcstd'][0]
    bil_colstd = params['bil_colstd'][0]
    bil_potts = params['bil_potts'][0]

    generate_segmentation(spc_std, spc_potts, bil_spcstd,
                          bil_colstd, bil_potts)
    return evaluate_segmentation(spc_std, spc_potts,
                                 bil_spcstd, bil_colstd, bil_potts)
