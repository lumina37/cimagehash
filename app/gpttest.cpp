#include <cstdint>
#include <intrin.h>

#include <bitset>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    auto input = cv::imread("sample.jpg");
    cv::Mat resizeImg, grayImg, bitsImg, outputArr;

    cv::resize(input, resizeImg, cv::Size(8, 8), 0, 0, cv::INTER_LINEAR_EXACT);
    if (input.channels() > 1)
        cv::cvtColor(resizeImg, grayImg, cv::COLOR_BGR2GRAY);
    else
        grayImg = resizeImg;

    uchar const imgMean = static_cast<uchar>(cvRound(cv::mean(grayImg)[0]));
    cv::compare(grayImg, imgMean, bitsImg, cv::CMP_GT);
    bitsImg /= 255;

    cv::Mat hash;
    hash.create(1, 8, CV_8U);
    uchar* hash_ptr = hash.ptr<uchar>(0);
    uchar const* bits_ptr = bitsImg.ptr<uchar>(0);
    std::bitset<8> bits;
    for (size_t i = 0, j = 0; i != bitsImg.total(); ++j) {
        for (size_t k = 0; k != 8; ++k) {
            // avoid warning C4800, casting do not work
            bits[k] = bits_ptr[i++] != 0;
        }
        hash_ptr[j] = static_cast<uchar>(bits.to_ulong());
    }
    std::cout << hash << std::endl;
}