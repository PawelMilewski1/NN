#include <vector>

#ifndef node_h
#define node_h

class node {
    public:
        std::vector<float> weights;
        float bias;
        float preActivationValue=0;
        float activationValue=0;
        float delta=0;
};

#endif //node_h