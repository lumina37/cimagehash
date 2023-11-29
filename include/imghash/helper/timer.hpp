#pragma once

#include <chrono>
#include <iostream>

class Timer
{
public:
    Timer() { start = std::chrono::high_resolution_clock::now(); }

    ~Timer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        double us = (double)duration * 0.001;

        std::cout << "Time cost:" << us << "us" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};