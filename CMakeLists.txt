cmake_minimum_required(VERSION 3.15)

project("cimghash" VERSION "0.1.0")

find_package(OpenCV COMPONENTS core imgcodecs imgproc img_hash)
set(IGH_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")

add_subdirectory(app)
