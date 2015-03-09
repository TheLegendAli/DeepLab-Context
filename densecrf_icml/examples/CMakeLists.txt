add_executable( densecrf_learning ppm.cpp lodepng.cpp common.cpp util.cpp densecrf_learning.cpp)
target_link_libraries( densecrf_learning densecrf optimization matio hdf5)

add_executable( dense_inference ppm.cpp common.cpp dense_inference.cpp )
target_link_libraries( dense_inference densecrf )

add_executable( dense_learning ppm.cpp common.cpp dense_learning.cpp )
target_link_libraries( dense_learning densecrf optimization )

add_executable( test_optimization test_optimization.cpp )
target_link_libraries( test_optimization optimization )

add_executable( test_permutohedral test_permutohedral.cpp )
target_link_libraries( test_permutohedral densecrf optimization )

add_executable( test_pairwise test_pairwise.cpp )
target_link_libraries( test_pairwise densecrf optimization )
