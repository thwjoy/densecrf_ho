
#include <iostream>

#include "runibfs.hpp"
#include "ibfs.h"

void gaussianFeatures(MatrixXf & features, int H, int W, float sx, float sy) {
    for( int j=0; j<H; j++ ) {
        for( int i=0; i<W; i++) {
            features(0,j*W+i) = i / sx;
            features(1,j*W+i) = j / sy;
        }
    }
}

void construct(IBFSGraph* g, MatrixXf & pairwise, int N, const MatrixXf & unaries, const MatrixXf & features, float weight) {

	g->initSize(N, N*N);

	// unary potentials
	for (int i = 0; i < N; ++i) {	// image grid
        g->addNode(i, unaries(1, i), unaries(0, i));
	}

	// binary potentials --> add edges
    pairwise.fill(0);
	for (int i = 0; i < N; ++i) {
		for (int j = i+1; j < N; ++j) {
            VectorXf featDiff = (features.col(i) - features.col(j));
            float w = 2 * weight * exp(-featDiff.squaredNorm());
            pairwise(i, j) = w;
            pairwise(j, i) = w;
            g->addEdge(i, j, w, w);
		}
	}
}

void minimumCut(VectorXs & labels, IBFSGraph* g, int N) {
	for (int i = 0; i < N; ++i) {	// image grid
        labels(i) = g->isNodeOnSrcSide(i, 1) ? 0 : 1;
	}
}

// should match with brute-force energy computation
double minimumEnergy(VectorXs & labels, const MatrixXf & unaries, const MatrixXf & pairwise, int N) {
    double e = 0;
    // unary 
    for (int i = 0; i < N; ++i) {
        e += unaries(labels(i), i);
    }
    // pairwise
    for (int i = 0; i < N; ++i) {
		for (int j = i+1; j < N; ++j) {
            e += pairwise(i, j) * abs(labels(i) - labels(j));
        }
    }
    return e;
}

double execute(IBFSGraph* g) {
	g->initGraph();
	g->computeMaxFlow();
	return g->getFlow();
}

double testIBFS(VectorXs & labels, int H, int W, const MatrixXf & unaries, float sigma, float weight) {

	IBFSGraph* g = new IBFSGraph(IBFSGraph::IB_INIT_FAST); // IB_INIT_COMPACT

    int N = W * H;
    MatrixXf features(2, N);
    gaussianFeatures(features, H, W, sigma, sigma);
    
    MatrixXf pairwise(N, N);
    construct(g, pairwise, N, unaries, features, weight);
    double flow = execute(g);
    
    minimumCut(labels, g, N);
    double energy = minimumEnergy(labels, unaries, pairwise, N);

    // flow may not be equal to energy: flow calculation is different ibfs!

    delete g;
    return energy;
}

