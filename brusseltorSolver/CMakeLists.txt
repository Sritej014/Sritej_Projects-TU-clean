cmake_minimum_required(VERSION 3.26)
project(Brusselator
    VERSION 0.1
    LANGUAGES CXX
    )

# Set C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Optimization and debug flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -g3")

# External Libraries
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# Include directories for libraries
include_directories(include)

# Add source files
set(SOURCES
    our_solver/brusselator.cpp
    our_solver/integrator.cpp
    our_solver/solver.cpp
)

# Add the executable
add_executable(brusselator ${SOURCES})

# link external libraries
target_link_libraries(brusselator 
    PRIVATE Eigen3::Eigen
    )

# installation rule
install(TARGETS brusselator DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
