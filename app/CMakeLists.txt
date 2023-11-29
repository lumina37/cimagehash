find_package(OpenCV COMPONENTS core imgcodecs imgproc img_hash REQUIRED)

function(igh_add_exec exec_name)
    add_executable(${exec_name} ${ARGV})
    target_include_directories(${exec_name} PRIVATE "../include" ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${exec_name} PRIVATE ${OpenCV_LIBS})
    target_compile_features(${exec_name} PRIVATE cxx_std_20)

    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${exec_name} PRIVATE "/arch:AVX2" "/Wall" "/Ox")
    else()
        target_compile_options(${exec_name} PRIVATE "-mavx2")
    endif()
endfunction()

igh_add_exec(main "main.cpp")
igh_add_exec(gpttest "gpttest.cpp")
