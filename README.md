## Synopsis

Codebase for the paper Efficient Relaxations for Dense CRFs with Sparse Higher Order Potentials, contains implemntations of QP and LP with and withouth higher order potentials. If using this code base please cite the relavant papers:

```

@article{joy2018efficient,
  title={Efficient Relaxations for Dense CRFs with Sparse Higher Order Potentials},
  author={Joy, Thomas and Desmaison, Alban and Ajanthan, Thalaiyasingam and Bunel, Rudy and Salzmann, Mathieu and Kohli, Pushmeet and Torr, Philip HS and Kumar, M Pawan},
  journal={arXiv preprint arXiv:1805.09028},
  year={2018}
}

@inproceedings{Ajanthan2016,
	author = {Ajanthan, Thalaiyasingam and Desmaison, Alban and Bunel, Rudy and Salzmann, Mathieu and Torr, Philip H. S. and Kumar, M. Pawan},
	booktitle = {Proceedings of the IEEE conference on Computer Vision and Pattern Recognition (CVPR)},
	title = {{Efficient Linear Programming for Dense CRFs}},
	year = {2017}
}

@inproceedings{Desmaison2016,
	author = {Desmaison, Alban and Bunel, Rudy and Kohli, Pushmeet and Torr, Philip H.S. and {Pawan Kumar}, M.},
	booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
	title = {{Efficient continuous relaxations for dense CRF}},
	year = {2016}
}

@inproceedings{Krahenbuhl2011,
	author = {Kr{\"{a}}hen, Philipp and Koltun, Vladlen},
	booktitle = {Proceedings of the conference on Neural Information Processing Systems (NIPS) },
	title = {{Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials}},
	year = {2011}
}


```

## Installation

Install in the standard way:

```
mkdir build
cd build
cmake ..
make -j
```

## Code Example

Once you have compiled the codebase, you should be able to run a small toy example by executing ./example_inference [algo_name] [params]. The possible algo_names are: qp, lp, qp_sp, and lp_sp. params should be given as sequence of numbers, with the first five correposponding to the pairwise parameters and the latter two corresponding to the higher order terms. The output segmentation is saved in the folder densecrf/data.

