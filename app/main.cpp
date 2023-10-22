#include <chrono>
#include <iostream>

#include <opencv2/img_hash.hpp>
#include <opencv2/imgcodecs.hpp>

#include "imghash/average_hash/avx2/impl.hpp"

int main()
{
    auto src1 = cv::imread("sample.jpg");
    auto src2 = src1.clone();
    std::cout << "src1 width: " << src1.cols << '\n';
    std::cout << "src1 height: " << src1.rows << '\n';
    std::cout << "src1 channels: " << src1.channels() << '\n';
    std::cout << "src1 step: " << src1.step << '\n';
    std::cout << '\n';

    /* My ahash */
    uint8_t res1[8];
    auto start = std::chrono::high_resolution_clock::now();
    igh::average::compute_ch3_div8(src1.data, src1.cols, src1.rows, (int)src1.step, res1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "My hash: ";
    for (uint8_t i : res1) {
        std::cout << (uint16_t)i << " ";
    }
    std::cout << '\n';
    std::cout << "time cost: " << (double)duration / 1000000 << " ms\n";

    /* OpenCV ahash */
    auto ptr_hasher = cv::img_hash::AverageHash::create();
    cv::Mat res2;
    start = std::chrono::high_resolution_clock::now();
    ptr_hasher->compute(src2, res2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "OpenCV hash: " << res2;
    std::cout << '\n';
    std::cout << "time cost: " << (double)duration / 1000000 << " ms" << std::endl;
}