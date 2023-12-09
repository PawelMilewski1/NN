/*
Pawel Milewski - Fall 2023
Main file to run the program
*/

#include <iostream>
#include "NN.h"

int main() {
    std::cout << "1. Train \n2. Test \n Enter '1' or '2':";
    char trainortest;
    std::cin >> trainortest;

    if (trainortest == '1') { // train
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
        double learningRate;
        std::cin >> learningRate;

        NeuralNetwork NeuralNet = NeuralNetwork();

        NeuralNet.initNN(NeuralNet, initialNN);
        NeuralNet.backPropLearning(trainingSet, NeuralNet, epochCount, learningRate);
        NeuralNet.createFile(NeuralNet, outputFile);
    } else if (trainortest == '2') { // test
        std::cout << "Neural Net:";
        std::string initialNN;
        std::cin >> initialNN;

        std::cout << "Testing Set:";
        std::string testingSet;
        std::cin >> testingSet;
        
        std::cout << "Output File:";
        std::string outputFile;
        std::cin >> outputFile;

        NeuralNetwork NeuralNet = NeuralNetwork();

        NeuralNet.initNN(NeuralNet, initialNN);
        NeuralNet.test(NeuralNet, testingSet, outputFile);
    } else { // test different hidden layer sizes, epoch counts and learning rates
        std::string initialNN = "concrete.init";
        std::string trainingSet = "concrete.train";
        std::string trainedNN = "o.txt";
        int epochCount = 0;
        double learningRate = 0;
        std::string testingSet = "concrete.test";
        std::string outputFile = "dummyfile.txt";

        std::vector<std::string> hiddenCount = {"concrete4.init", "concrete7.init", "concrete10.init", "concrete13.init", "concrete16.init", "concrete19.init", "concrete22.init", "concrete25.init", "concrete28.init", "concrete31.init"};
        std::vector<int> epochs = {100, 200, 300, 400, 500};
        std::vector<double> learningRates = {0.001, 0.005, 0.01, 0.05, 0.1};

        for (int i3 = 0; i3 < 10; i3++) {
            for (int i = 0; i < 5; i++) {
                for (int i2 = 0; i2 < 5; i2++) {
                    NeuralNetwork NeuralNet = NeuralNetwork();
                    NeuralNet.initNN(NeuralNet, hiddenCount[i3]);
                    NeuralNet.backPropLearning(trainingSet, NeuralNet, epochs[i], learningRates[i2]);
                    NeuralNet.createFile(NeuralNet, outputFile);
                    NeuralNetwork testingNet = NeuralNetwork();
                    NeuralNet.initNN(testingNet, outputFile);
                    std::cout << 3 * (i3 + 1) + 1 << " " << epochs[i] << " " << learningRates[i2] << " " << NeuralNet.test(testingNet, testingSet, outputFile) << std::endl;
                }
            }
        }
    }
}