include_directories( liblbfgs/include )
add_library( lbfgs liblbfgs/lib/lbfgs.c )

include_directories( ProbImage)
add_library( probimage ProbImage/probimage.cpp)

include_directories( ppm)
add_library( ppm ppm/ppm.cpp)
