import subprocess
import matlab.engine
import os

experiment_suffix = "qp_sp"
split_no = "1"
split = "val" + split_no
dataset = "MSRC"

def generate_segmentation(spc_std, spc_potts,
                          bil_spcstd, bil_colstd, bil_potts,
				const_1,const_2,const_3,
				norm_1, norm_2, norm_3):
    path_to_executable = 'build/alpha/cv-script'


    exp_path = '_'.join(map(str, [spc_std, spc_potts, bil_spcstd,
                                  bil_colstd, bil_potts,
				const_1,const_2,const_3,
				norm_1, norm_2, norm_3]))

    results_path = "/home/tomj/Documents/4YP/densecrf/data/CV" + split_no + "/" + dataset + "/" + experiment_suffix + "/" + exp_path
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
                     str(bil_spcstd), str(bil_colstd), str(bil_potts),
			str(const_1),str(const_2),str(const_3),
		 	str(norm_1), str(norm_2), str(norm_3)])
    

def evaluate_segmentation(spc_std, spc_potts,
                          bil_spcstd, bil_colstd, bil_potts,
				const_1,const_2,const_3,
				norm_1, norm_2, norm_3):
    eng = matlab.engine.start_matlab()
    exp_path = '_'.join(map(str, [spc_std, spc_potts, bil_spcstd,
                                  bil_colstd, bil_potts,
				const_1,const_2,const_3,
				norm_1, norm_2, norm_3]))
    path_to_results = "/home/tomj/Documents/4YP/densecrf/data/CV" + split_no + "/" + "MSRC/" + experiment_suffix + "/" + exp_path + "/" + experiment_suffix
    eng.addpath('/home/tomj/Documents/4YP/densecrf/tools/', nargout=0)
    ret = eng.CVmsrc_test(path_to_results, split)
    # This returns the value of the average accuracy.Spearmint
    # minimize things and we want to maximise this, so we should
    # return the negative
    print "params:\tspc_std:" + str(spc_std) + " , spc_potts:" + str(spc_potts) + " , bil_spcstd:" + str(bil_spcstd) + " , bil_colstd:" + str(bil_colstd) + " , bil_potts:" + str(bil_potts)
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
    const_1 = params['const_1'][0]
    const_2 = params['const_2'][0]
    const_3 = params['const_3'][0]
    norm_1 = params['norm_1'][0]
    norm_2 = params['norm_2'][0]
    norm_3 = params['norm_3'][0]

    generate_segmentation(spc_std, spc_potts, bil_spcstd,
                          bil_colstd, bil_potts,
				const_1,const_2,const_3,
				norm_1, norm_2, norm_3)
    return evaluate_segmentation(spc_std, spc_potts,
                                 bil_spcstd, bil_colstd, bil_potts,
				const_1,const_2,const_3,
				norm_1, norm_2, norm_3)
