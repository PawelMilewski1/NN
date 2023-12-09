/*
Pawel Milewski - Fall 2023
Definitions for functions for class "NeuralNetwork"
*/

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

double NeuralNetwork::test(NeuralNetwork& inputNN, std::string testingSet, std::string outputFileInput) {    
    std::ifstream inputFile(testingSet);

    std::string inputLine;
    std::getline(inputFile, inputLine);
    std::istringstream iss(inputLine);

    int numberofTests;
    std::vector<int> firstLine;
    int input;
    while (iss >> input) {
        firstLine.push_back(input);
    }
    numberofTests = firstLine[0];

    std::vector<std::vector<double>> ABCD(inputNN.outputSize, std::vector<double>(4, 0.0));

    double A = 0.0, B = 0.0, C = 0.0, D = 0.0;

    for (int i2 = 0; i2 < numberofTests; i2++) {

        std::string inputLine2;
        std::getline(inputFile, inputLine2);
        std::istringstream iss(inputLine2);

        double value;
        std::vector<double> outputs;
        std::vector<double> inputs;
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

        for (int i3 = 0; i3< inputNN.outputSize; i3++) {
            if (inputNN.neuralNet[2][i3].activationValue >= 0.5 && outputs[i3] == 1) {
                ABCD[i3][0]+=1.0;
                A+=1.0;
            } else if (inputNN.neuralNet[2][i3].activationValue >= 0.5 && outputs[i3] == 0) {
                ABCD[i3][1]+=1.0;
                B+=1.0;
            } else if (inputNN.neuralNet[2][i3].activationValue < 0.5 && outputs[i3] == 1) {
                ABCD[i3][2]+=1.0;
                C+=1.0;
            } else if (inputNN.neuralNet[2][i3].activationValue < 0.5 && outputs[i3] == 0) {
                ABCD[i3][3]+=1.0;
                D+=1.0;
            } 
        }
    }



    double overallAccuracy = 0.0, precision = 0.0, recall = 0.0, f1 = 0.0;
    double totaloverallAccuracy = 0.0, totalprecision = 0.0, totalrecall = 0.0;

    std::ofstream outputFile(outputFileInput);

    for (int i = 0; i < inputNN.outputSize; i++) {
        outputFile << std::fixed << std::setprecision(0) << ABCD[i][0] << " " << ABCD[i][1] << " " << ABCD[i][2] << " " << ABCD[i][3] << " ";
        overallAccuracy = (ABCD[i][0] + ABCD[i][3]) / (ABCD[i][0] + ABCD[i][1] + ABCD[i][2] + ABCD[i][3]);
        totaloverallAccuracy += overallAccuracy;        
        precision = ABCD[i][0] / (ABCD[i][0] + ABCD[i][1]);
        totalprecision += precision;
        recall = ABCD[i][0] / (ABCD[i][0] + ABCD[i][2]);
        totalrecall += recall;
        outputFile << std::fixed << std::setprecision(3) << overallAccuracy << " ";
        outputFile << std::fixed << std::setprecision(3) << precision << " ";
        outputFile << std::fixed << std::setprecision(3) << recall << " ";
        outputFile << std::fixed << std::setprecision(3) << (2 * precision * recall) / (precision + recall) << "\n";
    }


    overallAccuracy = (A + D) / (A + B + C + D);
    precision = A / (A + B);
    recall = A / (A + C);
    f1 = (2 * precision * recall) / (precision + recall);

    outputFile << std::fixed << std::setprecision(3) << overallAccuracy << " ";
    outputFile << std::fixed << std::setprecision(3) << precision << " ";
    outputFile << std::fixed << std::setprecision(3) << recall << " ";
    outputFile << std::fixed << std::setprecision(3) << (2 * precision * recall) / (precision + recall) << "\n";
    
    outputFile << std::fixed << std::setprecision(3) << totaloverallAccuracy / inputNN.outputSize << " ";
    outputFile << std::fixed << std::setprecision(3) << totalprecision / inputNN.outputSize<< " ";
    outputFile << std::fixed << std::setprecision(3) << totalrecall / inputNN.outputSize << " ";
    outputFile << std::fixed << std::setprecision(3) << (2 * totalprecision / inputNN.outputSize * totalrecall / inputNN.outputSize) / (totalprecision /inputNN.outputSize + totalrecall / inputNN.outputSize) << "\n";

    outputFile.close();

    return f1;
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
        double weight;
        std::vector<double> nodeWeights;
        while (iss >> weight) {
            nodeWeights.push_back(weight);
        }

        inputNN.neuralNet[1][i].bias = nodeWeights[0];
        inputNN.neuralNet[1][i].weights = std::vector<double>(nodeWeights.begin()+1, nodeWeights.end());
    }

    for (int i = 0; i < inputNN.outputSize; i++) {
        std::string inputLine3;
        std::getline(inputFile, inputLine3);
        std::istringstream iss(inputLine3);
        double weight;
        std::vector<double> nodeWeights;
        while (iss >> weight) {
            nodeWeights.push_back(weight);
        }

        inputNN.neuralNet[2][i].bias = nodeWeights[0];
        inputNN.neuralNet[2][i].weights = std::vector<double>(nodeWeights.begin()+1, nodeWeights.end());
    }
    inputFile.close();

    inputNN.createFile(inputNN, "input.txt");
}

NeuralNetwork NeuralNetwork::backPropLearning(std::string trainingSet, NeuralNetwork& inputNN, int epochCount, double learningRate) {
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

            double value;
            std::vector<double> outputs;
            std::vector<double> inputs;
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
                double sum = 0;
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
        
        inputFile.close();
    }

    return inputNN;
}

void NeuralNetwork::createFile(NeuralNetwork& inputNN, std::string outputFileInput) {
    std::ofstream outputFile(outputFileInput);
    outputFile << inputNN.inputSize << " " << inputNN.hiddenSize << " " << inputNN.outputSize << "\n";
    for (int i = 0; i < inputNN.hiddenSize; i++) {
        outputFile << std::fixed << std::setprecision(3) << inputNN.neuralNet[1][i].bias << " ";
        for (int i2 = 0; i2 < inputNN.inputSize; i2++) {
            if (i2 == inputNN.inputSize - 1) {
                outputFile << std::fixed << std::setprecision(3) << inputNN.neuralNet[1][i].weights[i2] << "\n";
            } else {
                outputFile << std::fixed << std::setprecision(3) << inputNN.neuralNet[1][i].weights[i2] << " ";
            }
        }
    }

    for (int i = 0; i < inputNN.outputSize; i++) {
        outputFile << std::fixed << std::setprecision(3) << inputNN.neuralNet[2][i].bias << " ";
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

double NeuralNetwork::sigmoid(double x) {
    return 1/(1+std::exp(-x));
}

double NeuralNetwork::sigmoidPrime(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}
