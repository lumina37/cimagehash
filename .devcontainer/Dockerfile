FROM debian:12

ARG OPENCV_VERSION="4.9.0"
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends git cmake g++ wget \
    && wget -O opencv.tar.gz https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz \
    && wget -O opencv_contrib.tar.gz https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.tar.gz \
    && tar -xzf opencv.tar.gz \
    && tar -xzf opencv_contrib.tar.gz \
    && cd opencv-${OPENCV_VERSION} && mkdir build \
    && cmake -S . -B build -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules \
    -DBUILD_JAVA=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_opencv_calib3d=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DBUILD_opencv_gapi=OFF \
    -DBUILD_opencv_java_bindings_generator=OFF \
    -DBUILD_opencv_js=OFF \
    -DBUILD_opencv_js_bindings_generator=OFF \
    -DBUILD_opencv_ml=OFF \
    -DBUILD_opencv_objc_bindings_generator=OFF \
    -DBUILD_opencv_objdetect=OFF \
    -DBUILD_opencv_python_bindings_generator=OFF \
    -DBUILD_opencv_python_tests=OFF \
    -DBUILD_opencv_aruco=OFF \
    -DBUILD_opencv_bgsegm=OFF \
    -DBUILD_opencv_bioinspired=OFF \
    -DBUILD_opencv_ccalib=OFF \
    -DBUILD_opencv_dnn_objdetect=OFF \
    -DBUILD_opencv_dnn_superres=OFF \
    -DBUILD_opencv_dpm=OFF \
    -DBUILD_opencv_face=OFF \
    -DBUILD_opencv_fuzzy=OFF \
    -DBUILD_opencv_hfs=OFF \
    -DBUILD_opencv_intensity_transform=OFF \
    -DBUILD_opencv_line_descriptor=OFF \
    -DBUILD_opencv_mcc=OFF \
    -DBUILD_opencv_optflow=OFF \
    -DBUILD_opencv_plot=OFF \
    -DBUILD_opencv_rapid=OFF \
    -DBUILD_opencv_reg=OFF \
    -DBUILD_opencv_rgbd=OFF \
    -DBUILD_opencv_saliency=OFF \
    -DBUILD_opencv_shape=OFF \
    -DBUILD_opencv_stereo=OFF \
    -DBUILD_opencv_structured_light=OFF \
    -DBUILD_opencv_superres=OFF \
    -DBUILD_opencv_surface_matching=OFF \
    -DBUILD_opencv_text=OFF \
    -DBUILD_opencv_tracking=OFF \
    && cmake --build build \
    && cd build && make install
