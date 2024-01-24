find_package(OpenCV COMPONENTS core imgcodecs imgproc img_hash)

function(igh_add_exec exec_name)
    add_executable(${exec_name} ${ARGV})
    target_include_directories(${exec_name} PRIVATE "../include" ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${exec_name} PRIVATE ${OpenCV_LIBS})
    target_compile_features(${exec_name} PRIVATE cxx_std_20)

    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${exec_name} PRIVATE "/Wall")
    else()
        target_compile_options(${exec_name} PRIVATE "-Wall" "-Wextra")
    endif()
endfunction()

igh_add_exec(bench_this "bench_this.cpp")
igh_add_exec(bench_ocv "bench_ocv.cpp")
igh_add_exec(gpttest "gpttest.cpp")
