#include <vector>
#include <string>

#include "node.h"

#ifndef NN_H
#define NN_H

class NeuralNetwork {
    public:
        NeuralNetwork();
        std::vector<std::vector<node>> neuralNet;
        int inputSize;
        int hiddenSize;
        int outputSize;
        void initNN(NeuralNetwork& inputNN, std::string initialNN);
        NeuralNetwork backPropLearning(std::string trainingSet, NeuralNetwork& inputNN, int epochCount, float learningRate);
        void createFile(NeuralNetwork& inputNN, std::string outputFileInput);
        float sigmoid(float x);
        float sigmoidPrime(float x);
};

#endif // NN_H