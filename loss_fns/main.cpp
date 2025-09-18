#include <iostream>
#include "Losses.h"

int main() {
    float xi = 0.45;
    float yi = 0.6;

    Losses loss;

    std::cout << "Mean Absolute Error : " << loss.mae(yi, xi) << std::endl;
    std::cout << "Binary Cross Entropy : " << loss.bce(yi, xi) << std::endl;

    return 0;
}

