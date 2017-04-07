import subprocess
import matlab.engine
import os

experiment_suffix = "qp_sp_100"
split = "cv1/val"
dataset = "MSRC"

def generate_segmentation(spc_std, spc_potts,
                          bil_spcstd, bil_colstd, bil_potts):
    path_to_executable = 'build/alpha/cv-script'
    exp_path = '_'.join(map(str, [spc_std, spc_potts, bil_spcstd,
                                  bil_colstd, bil_potts]))

    results_path = "/home/tomj/Documents/4YP/densecrf/data/CV/" + dataset + "/" + experiment_suffix + "/" + exp_path
    try:
        os.makedirs(results_path)
    except OSError:
        pass
    os.chdir("/home/tomj/Documents/4YP/densecrf")
    subprocess.call([path_to_executable,
                     split,
                     dataset,
                     experiment_suffix,
                     results_path,
                     str(spc_std), str(spc_potts),
                     str(bil_spcstd), str(bil_colstd), str(bil_potts)])
    

def evaluate_segmentation(spc_std, spc_potts,
                          bil_spcstd, bil_colstd, bil_potts):
    eng = matlab.engine.start_matlab()
    exp_path = '_'.join(map(str, [spc_std, spc_potts, bil_spcstd,
                                  bil_colstd, bil_potts]))
    path_to_results = "/home/tomj/Documents/4YP/densecrf/data/CV/MSRC/" + experiment_suffix + "/" + exp_path + "/" + experiment_suffix
    eng.addpath('/home/tomj/Documents/4YP/densecrf/tools/', nargout=0)
    ret = eng.CVmsrc_test(path_to_results, split)
    # This returns the value of the average accuracy.Spearmint
    # minimize things and we want to maximise this, so we should
    # return the negative
    txt_results_file = path_to_results + '/results.txt'
    with open(txt_results_file, 'w') as f:
        f.write(exp_path)
        f.write('/n')
        f.write(str(ret))
    return -ret


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
