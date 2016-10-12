import subprocess
import sys
import os
import glob

experiment_suffix = "prox_lp"


def run_inference(prox_max_iter, fw_max_iter, prox_reg_const, dual_gap_tol, qp_tol):
    # DC-neg params
    spc_std = 3
    spc_potts = 0.5
    bil_spcstd = 50
    bil_colstd = 1
    bil_potts = 1
    qp_max_iter = 1000
    best_int = 1

    path_to_executable = '/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/lp_densecrf/densecrf/build/alpha/cv-script-sp'
    split = "Validation"
    dataset = "MSRC_2"

    exp_path = '/'.join(map(str, [prox_max_iter, fw_max_iter, prox_reg_const, dual_gap_tol, qp_tol]))

    results_path = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/CV/" + experiment_suffix + "/" + exp_path + "/"
    try:
        os.makedirs(results_path)
    except OSError:
        pass
    subprocess.call([path_to_executable,
                     split,
                     dataset,
                     experiment_suffix,
                     results_path,
                     str(spc_std), str(spc_potts), str(bil_spcstd), str(bil_colstd), str(bil_potts),
                     str(prox_max_iter), str(fw_max_iter), str(qp_max_iter), str(prox_reg_const), 
                     str(dual_gap_tol), str(qp_tol), str(best_int)])
#    os.system(path_to_executable + " " +
#                     split + " " +
#                     dataset + " " +
#                     experiment_suffix + " " +
#                     results_path + " " +
#                     str(spc_std)+ " " + str(spc_potts) + " " + str(bil_spcstd) + " " + str(bil_colstd) + " " + str(bil_potts) + " " +
#                     str(prox_max_iter) + " " + str(fw_max_iter) + " " + str(qp_max_iter) + " " + str(prox_reg_const) + " " + 
#                     str(dual_gap_tol) + " " + str(qp_tol) + " " + str(best_int))

def compute_avg_energy_time(folder_path):
    all_txts = glob.glob(folder_path + '*.txt')
    energy = []
    time = []
    for txt_file in all_txts:
        with open(txt_file, 'r') as txt:
            content = txt.read().strip().split('\t')
            time.append(float(content[0]))
            energy.append(float(content[2]))    # integral-energy

    # average
    energy_time = []
    energy_time.append(sum(energy)/float(len(energy)))
    energy_time.append(sum(time)/float(len(time)))
    return energy_time


def get_energy_time(prox_max_iter, fw_max_iter, prox_reg_const, dual_gap_tol, qp_tol):

    exp_path = '/'.join(map(str, [prox_max_iter, fw_max_iter, prox_reg_const, dual_gap_tol, qp_tol]))
    path_to_results = "/media/ajanthan/b7391340-f7ed-49ef-9dab-f3749bde5917/ajanthan/NICTA/Research/ubuntu_codes/data/CV/" + experiment_suffix + "/" + exp_path + "/" + experiment_suffix

    avg_energy_time = compute_avg_energy_time(path_to_results + "/")

    ret = avg_energy_time[0] + avg_energy_time[1] * 1000    # bring energy and time to the same scale

    # This returns the value of the average energy and time combined. Spearmint
    # minimize things
    txt_results_file = path_to_results + '/results.txt'
    with open(txt_results_file, 'w') as f:
        f.write(exp_path)
        f.write('/n')
        f.write(str(avg_energy_time) + " " + str(ret))
    return ret


# Write a function like this called 'main'
def main(job_id, params):
    prox_max_iter = params['prox_max_iter'][0]
    fw_max_iter = params['fw_max_iter'][0]
    prox_reg_const = params['prox_reg_const'][0]
    dual_gap_tol = params['dual_gap_tol'][0]
    qp_tol = params['qp_tol'][0]

    run_inference(prox_max_iter, fw_max_iter, prox_reg_const, dual_gap_tol, qp_tol)

    return get_energy_time(prox_max_iter, fw_max_iter, prox_reg_const, dual_gap_tol, qp_tol)

