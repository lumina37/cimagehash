cmake_minimum_required(VERSION 3.15)

project(cimghash VERSION "0.1.0" LANGUAGES CXX)

include(cmake/options.cmake)
include(cmake/dependencies.cmake)
include(cmake/helper.cmake)

add_subdirectory(bin)
