cmake_minimum_required(VERSION 2.8)

project( MultiplicacionMatricial )
enable_language(CUDA)

include_directories( ${CUDA_INCLUDE_DIRS} )
add_executable( MultiplicacionCuda MultiplicacionCuda.cu )
target_link_libraries( MultiplicacionCuda ${CUDA_LIBS})