cmake_minimum_required (VERSION 2.6)
project(densecrf)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include_directories( include )
# Jay add link directories
link_directories(/home/lcchen/workspace/ext/hdf5-1.8.13/lib)

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (external)
