find_package(Catch2 3)
find_package(OpenCV COMPONENTS core imgcodecs)

add_executable(test "test_simd.cpp")
target_include_directories(test PRIVATE "../include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test PRIVATE ${OpenCV_LIBS} Catch2::Catch2WithMain)
target_compile_features(test PRIVATE cxx_std_20)

include(CTest)
include(Catch)
catch_discover_tests(test)
