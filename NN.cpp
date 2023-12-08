#include "NN.h"
#include "node.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <cmath>

NeuralNetwork::NeuralNetwork() {

}

void NeuralNetwork::initNN(NeuralNetwork& inputNN, std::string initialNN) {
    std::ifstream inputFile(initialNN);

    std::string inputLine;
    std::getline(inputFile, inputLine);
    std::istringstream iss(inputLine);
    int layerSize;
    std::vector<int> layerSizes;
    while (iss >> layerSize) {
        layerSizes.push_back(layerSize);
    }

    inputNN.inputSize = layerSizes[0];
    inputNN.hiddenSize = layerSizes[1];
    inputNN.outputSize = layerSizes[2];

    inputNN.neuralNet.resize(3); {
        inputNN.neuralNet[0].resize(inputNN.inputSize);
        inputNN.neuralNet[1].resize(inputNN.hiddenSize);
        inputNN.neuralNet[2].resize(inputNN.outputSize);
    }

    for (int i = 0; i < inputNN.hiddenSize; i++) {
        std::string inputLine2;
        std::getline(inputFile, inputLine2);
        std::istringstream iss(inputLine2);
        float weight;
        std::vector<float> nodeWeights;
        while (iss >> weight) {
            nodeWeights.push_back(weight);
        }

        inputNN.neuralNet[1][i].bias = nodeWeights[0];
        inputNN.neuralNet[1][i].weights = std::vector<float>(nodeWeights.begin()+1, nodeWeights.end());
    }

    for (int i = 0; i < inputNN.outputSize; i++) {
        std::string inputLine3;
        std::getline(inputFile, inputLine3);
        std::istringstream iss(inputLine3);
        float weight;
        std::vector<float> nodeWeights;
        while (iss >> weight) {
            nodeWeights.push_back(weight);
        }

        inputNN.neuralNet[2][i].bias = nodeWeights[0];
        inputNN.neuralNet[2][i].weights = std::vector<float>(nodeWeights.begin()+1, nodeWeights.end());
    }
    inputFile.close();

    inputNN.createFile(inputNN, "input.txt");
}

NeuralNetwork NeuralNetwork::backPropLearning(std::string trainingSet, NeuralNetwork& inputNN, int epochCount, float learningRate) {
    std::ifstream inputFile(trainingSet);

    std::string inputLine;
    std::getline(inputFile, inputLine);
    std::istringstream iss(inputLine);

    int numberofExamples;
    std::vector<int> firstLine;
    int input;
    while (iss >> input) {
        firstLine.push_back(input);
    }
    numberofExamples = firstLine[0];

    inputFile.close();

    for (int i = 0; i < epochCount; i++) {
        inputFile.open(trainingSet);
        std::getline(inputFile, inputLine);
        for (int i2 = 0; i2 < numberofExamples; i2++) {

            std::string inputLine2;
            std::getline(inputFile, inputLine2);
            std::istringstream iss(inputLine2);

            float value;
            std::vector<float> outputs;
            std::vector<float> inputs;
            int counter = 0;
            while (iss >> value) {
                if (counter < inputNN.inputSize) {
                    inputs.push_back(value);
                    counter++;
                } else {
                    outputs.push_back(value);
                }
            }

            for (int i3= 0; i3 < inputNN.inputSize; i3++) {
                inputNN.neuralNet[0][i3].activationValue = inputs[i3];
            }

            for (int i3 = 0; i3 < inputNN.hiddenSize; i3++) {
                inputNN.neuralNet[1][i3].preActivationValue = inputNN.neuralNet[1][i3].bias * -1;
                for (int i4= 0; i4 < inputNN.inputSize; i4++) {
                    inputNN.neuralNet[1][i3].preActivationValue += inputNN.neuralNet[1][i3].weights[i4] * inputNN.neuralNet[0][i4].activationValue;
                }
                inputNN.neuralNet[1][i3].activationValue = inputNN.sigmoid(inputNN.neuralNet[1][i3].preActivationValue);
            }

            for (int i3 = 0; i3 < inputNN.outputSize; i3++) {
                inputNN.neuralNet[2][i3].preActivationValue = inputNN.neuralNet[2][i3].bias * -1;
                for (int i4= 0; i4 < inputNN.hiddenSize; i4++) {
                    inputNN.neuralNet[2][i3].preActivationValue += inputNN.neuralNet[2][i3].weights[i4] * inputNN.neuralNet[1][i4].activationValue;
                }
                inputNN.neuralNet[2][i3].activationValue = inputNN.sigmoid(inputNN.neuralNet[2][i3].preActivationValue);              
            }

            for (int i3 = 0; i3 < inputNN.outputSize; i3++) {
                inputNN.neuralNet[2][i3].delta = inputNN.sigmoidPrime(inputNN.neuralNet[2][i3].preActivationValue) * (outputs[i3] - inputNN.neuralNet[2][i3].activationValue);
            }

            for (int i3 = 0; i3 < inputNN.hiddenSize; i3++) {
                float sum = 0;
                for (int i4 = 0; i4 < inputNN.outputSize; i4++) {
                    sum = sum + inputNN.neuralNet[2][i4].weights[i3] * inputNN.neuralNet[2][i4].delta;
                }
                inputNN.neuralNet[1][i3].delta = inputNN.sigmoidPrime(inputNN.neuralNet[1][i3].preActivationValue) * sum;
            }

            for (int output = 0; output < inputNN.outputSize; output++) {
                for (int outputinput = 0; outputinput < inputNN.hiddenSize; outputinput++) {
                    inputNN.neuralNet[2][output].weights[outputinput] += learningRate * inputNN.neuralNet[1][outputinput].activationValue * inputNN.neuralNet[2][output].delta;
                }
                inputNN.neuralNet[2][output].bias = inputNN.neuralNet[2][output].bias + learningRate * -1 * inputNN.neuralNet[2][output].delta;
            }

            for (int hidden = 0; hidden < inputNN.hiddenSize; hidden++) {
                for (int hiddeninput = 0; hiddeninput < inputNN.inputSize; hiddeninput++) {
                    inputNN.neuralNet[1][hidden].weights[hiddeninput] += learningRate * inputNN.neuralNet[0][hiddeninput].activationValue * inputNN.neuralNet[1][hidden].delta;
                }
                inputNN.neuralNet[1][hidden].bias = inputNN.neuralNet[1][hidden].bias + learningRate * -1 * inputNN.neuralNet[1][hidden].delta;
            }

            for (int i3 = 0; i3 < inputNN.inputSize; i3++) {
                inputNN.neuralNet[0][i3].preActivationValue = 0;
                inputNN.neuralNet[0][i3].activationValue = 0;
                inputNN.neuralNet[0][i3].delta = 0;
            }
            for (int i3 = 0; i3 < inputNN.hiddenSize; i3++) {
                inputNN.neuralNet[1][i3].preActivationValue = 0;
                inputNN.neuralNet[1][i3].activationValue = 0;
                inputNN.neuralNet[1][i3].delta = 0;
            }
            for (int i3 = 0; i3 < inputNN.outputSize; i3++) {
                inputNN.neuralNet[2][i3].preActivationValue = 0;
                inputNN.neuralNet[2][i3].activationValue = 0;
                inputNN.neuralNet[2][i3].delta = 0;
            }
        }
        
        std::cout << "epoCh:" << i << std::endl << std::endl;
        inputFile.close();
    }

    return inputNN;
}

void NeuralNetwork::createFile(NeuralNetwork& inputNN, std::string outputFileInput) {
    std::ofstream outputFile(outputFileInput);
    outputFile << inputNN.inputSize << " " << inputNN.hiddenSize << " " << inputNN.outputSize;
    for (int i = 0; i < inputNN.hiddenSize; i++) {
        outputFile << "\n" << std::fixed << std::setprecision(3) << inputNN.neuralNet[1][i].bias << " ";
        for (int i2 = 0; i2 < inputNN.inputSize; i2++) {
            if (i2 == inputNN.inputSize - 1) {
                outputFile << std::fixed << std::setprecision(3) << inputNN.neuralNet[1][i].weights[i2];
            } else {
                outputFile << std::fixed << std::setprecision(3) << inputNN.neuralNet[1][i].weights[i2] << " ";
            }
        }
    }

    for (int i = 0; i < inputNN.outputSize; i++) {
        outputFile << "\n" << std::fixed << std::setprecision(3) << inputNN.neuralNet[2][i].bias << " ";
        for (int i2 = 0; i2 < inputNN.hiddenSize; i2++) {
            if (i2 == inputNN.hiddenSize - 1) {
                outputFile << std::fixed << std::setprecision(3) << inputNN.neuralNet[2][i].weights[i2] << "\n";
            } else {
                outputFile << std::fixed << std::setprecision(3) << inputNN.neuralNet[2][i].weights[i2] << " ";
            }
        }
    }

    outputFile.close();
}

float NeuralNetwork::sigmoid(float x) {
    return 1/(1+std::exp(-x));
}

float NeuralNetwork::sigmoidPrime(float x) {
    float sig = sigmoid(x);
    return sig * (1 - sig);
}
