#include <iostream>
#include "NN.h"

int main() {
    std::cout << "1. Train \n2. Test \n Enter '1' or '2':";
    int trainortest;
    std::cin >> trainortest;

    if (trainortest == 1) { // train
        std::cout << "Initial Neural Net:";
        std::string initialNN;
        std::cin >> initialNN;

        std::cout << "Training Set:";
        std::string trainingSet;
        std::cin >> trainingSet;
        
        std::cout << "Output File:";
        std::string outputFile;
        std::cin >> outputFile;
        
        std::cout << "Epoch Count:";
        int epochCount;
        std::cin >> epochCount;
        
        std::cout << "Learning Rate:";
        float learningRate;
        std::cin >> learningRate;

        NeuralNetwork NeuralNet = NeuralNetwork();

        NeuralNet.initNN(NeuralNet, initialNN);
        NeuralNet.backPropLearning(trainingSet, NeuralNet, epochCount, learningRate);
        NeuralNet.createFile(NeuralNet, outputFile);
    }
}