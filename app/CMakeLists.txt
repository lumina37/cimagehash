find_package(OpenCV COMPONENTS core imgcodecs imgproc img_hash REQUIRED)

add_executable(main "main.cpp")
target_include_directories(main PRIVATE "../include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
target_compile_features(main PRIVATE cxx_std_20)

add_executable(gpttest "gpttest.cpp")
target_include_directories(gpttest PRIVATE "../include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gpttest PRIVATE ${OpenCV_LIBS})
target_compile_features(gpttest PRIVATE cxx_std_20)
