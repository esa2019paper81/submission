cmake_minimum_required(VERSION 3.5)
project(Puffinn CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -Wno-noexcept-type -march=native -O3 -g -fopenmp")

include_directories("include" "libs")

file(GLOB ExampleFiles "examples/*.cpp")
add_library(dummy OBJECT ${ExampleFiles})

include_directories("test/include")
add_executable(Test "test/main.cpp" "test/code.cpp")
target_link_libraries(Test)
