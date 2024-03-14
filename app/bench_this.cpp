#include <chrono>
#include <iostream>

#include <opencv2/img_hash.hpp>
#include <opencv2/imgcodecs.hpp>

#include "imghash/average_hash/generic/impl.hpp"

int main()
{
    auto src = cv::imread("sample.jpg");
    std::cout << "src width: " << src.cols << '\n';
    std::cout << "src height: " << src.rows << '\n';
    std::cout << "src channels: " << src.channels() << '\n';
    std::cout << "src step: " << src.step << '\n';
    std::cout << '\n';

    /* My ahash */
    uint8_t res[8];
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        igh::ahash::compute_ch3_div8(src.data, src.cols, src.rows, src.step, res);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "My hash: ";
    for (uint8_t i : res) {
        std::cout << (uint16_t)i << " ";
    }
    std::cout << '\n';
    std::cout << "time cost: " << (double)duration / 1000000 << " ms\n";
}