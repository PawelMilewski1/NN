/*
Pawel Milewski - Fall 2023
Variables for class "node"
*/

#include <vector>

#ifndef node_h
#define node_h

class node {
    public:
        std::vector<double> weights;
        double bias;
        double preActivationValue=0;
        double activationValue=0;
        double delta=0;
};

#endif //node_h