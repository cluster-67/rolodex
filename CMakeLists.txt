cmake_minimum_required(VERSION 3.14)
project(project67)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
# On Cray systems, this is typically provided by loading the `cray-hdf5` module.
find_package(HDF5 REQUIRED COMPONENTS CXX)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(knn
    src/knn_main.cpp
    src/dataset.cpp
    src/kmeans.cpp
)
target_link_libraries(knn PRIVATE HDF5::HDF5)
target_include_directories(knn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
)
target_compile_options(knn PRIVATE -O3 -march=native -funroll-loops)

add_executable(knn_openmp
    src/knn_openmp_main.cpp
    src/dataset.cpp
    src/kmeans.cpp
    src/kmeans_openmp.cpp
)
target_link_libraries(knn_openmp PRIVATE HDF5::HDF5 OpenMP::OpenMP_CXX)
target_include_directories(knn_openmp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
)

add_executable(knn_mpi
    src/knn_mpi_main.cpp
    src/dataset.cpp
    src/kmeans.cpp
    src/kmeans_mpi.cpp
)
target_link_libraries(knn_mpi PRIVATE HDF5::HDF5 MPI::MPI_CXX)
target_include_directories(knn_mpi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
)
