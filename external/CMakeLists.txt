include_directories( liblbfgs/include )
add_library( lbfgs liblbfgs/lib/lbfgs.c )

include_directories( ProbImage)
add_library( probimage ProbImage/probimage.cpp)

include_directories( ibfs )
add_library( ibfs ibfs/ibfs.cpp )
