#include <chrono>
#include <iostream>

#include <opencv2/img_hash.hpp>
#include <opencv2/imgcodecs.hpp>

#include "imghash/average_hash/avx2/impl.hpp"

int main()
{
    auto src = cv::imread("sample.jpg");
    std::cout << "src width: " << src.cols << '\n';
    std::cout << "src height: " << src.rows << '\n';
    std::cout << "src channels: " << src.channels() << '\n';
    std::cout << "src step: " << src.step << '\n';
    std::cout << '\n';

    /* OpenCV ahash */
    cv::Mat res2;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        auto ptr_hasher = cv::img_hash::AverageHash::create();
        ptr_hasher->compute(src, res2);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "OpenCV hash: " << res2;
    std::cout << '\n';
    std::cout << "time cost: " << (double)duration / 1000000 << " ms" << std::endl;
}