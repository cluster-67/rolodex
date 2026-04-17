cmake_minimum_required(VERSION 3.14)
project(project67)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)
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

add_executable(knn knn.cpp)
target_link_libraries(knn PRIVATE HDF5::HDF5)
target_include_directories(knn PRIVATE ${HDF5_INCLUDE_DIRS})

# add_executable(serial main.cpp serial.cpp common.h)

# add_executable(openmp main.cpp openmp.cpp common.h)
# target_link_libraries(openmp PRIVATE OpenMP::OpenMP_CXX)

# Copy the job scripts
# configure_file(job-serial job-serial COPYONLY)
# configure_file(job-openmp job-openmp COPYONLY)
