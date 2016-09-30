
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

void construct(IBFSGraph* g, int N, const MatrixXf & unaries, const MatrixXf & features, float weight) {

	g->initSize(N, N*N);

	// unary potentials
	for (int i = 0; i < N; ++i) {	// image grid
        g->addNode(i, unaries(1, i), unaries(0, i));
	}

    MatrixXf pweights(N, N);
    pweights.fill(0);
	// binary potentials --> accumulate weights
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            VectorXf featDiff = (features.col(i) - features.col(j));
            pweights(i, j) += weight * exp(-featDiff.squaredNorm());
		}
	}
    // add edges
    for (int i = 0; i < N; ++i) {
		for (int j = i+1; j < N; ++j) {
            float w = pweights(i, j) + pweights(j, i);
            g->addEdge(i, j, w, w);
		}
	}
}

void minimumCut(VectorXs & labels, IBFSGraph* g, int N) {
	for (int i = 0; i < N; ++i) {	// image grid
        labels(i) = g->isNodeOnSrcSide(i, 1) ? 0 : 1;
	}
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
        
    construct(g, N, unaries, features, weight);
    double flow = execute(g);
    
    minimumCut(labels, g, N);

    delete g;
	return flow;
}

