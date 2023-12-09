Pawel Milewski
AI Fall 2023
Neural Network Submission

Dataset: Cement Manufacturing (https://www.kaggle.com/datasets/vinayakshanawad/cement-manufacturing-concrete-dataset/)

From the site:
    Data Description
    The actual concrete compressive strength (MPa) for a given mixture under a
    specific age (days) was determined from laboratory. Data is in raw form (not scaled). The data has 8 quantitative input variables, and 1 quantitative output variable, and 1030 instances (observations).

    Domain
    Cement manufacturing

    Context
    Concrete is the most important material in civil engineering. The concrete compressive strength is a highly nonlinear function of age and ingredients. These ingredients include cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggregate.

    Attribute Information
    Cement : measured in kg in a m3 mixture
    Blast : measured in kg in a m3 mixture
    Fly ash : measured in kg in a m3 mixture
    Water : measured in kg in a m3 mixture
    Superplasticizer : measured in kg in a m3 mixture
    Coarse Aggregate : measured in kg in a m3 mixture
    Fine Aggregate : measured in kg in a m3 mixture
    Age : day (1~365)
    Concrete compressive strength measured in MPa

Training set file: concrete.train
Testing set file: concrete.test 

Input is amount of cement, blast, fly ash, water, superplasticizer, coarse aggregate, fine aggregate (these are all in kg)
Another input is the age (most likely time given for curing of concrete)
There is one output which is concrete compressive strength
    This output was simplied (converted into a boolean). This was done by using a range for the concrete compressive strength. Concrete compressive strength can be categorized into weak, medium and strong concrete. For the purpose of this project, the output (now a boolean) was determined based on whether or not the concrete is considered strong or not. A strong concrete typically has a psi greater than 5000 (just some additional info: a weak concrete is below 2500 psi)
    Therefore, concrete compressive strength >= 5000 psi -> output of 1
                                              < 5000 psi -> output of 0
                                              (Note that I converted and used MPa in the excel sheet)

Reasonable learning was dependent on how close the resulting F1 value was to 1 (therefore which combination of hidden layer size, learning rate and epoch count lead to the largest possible F1 value)
A combination of the following values for hidden layer size, epoch count and learning rate were used:
    Hidden Layer Sizes: 4 7 10 13 16 19 22 25 28 31
    Epoch Counts: 100 200 300 400 500
    Learning Rates: 0.001 0.05 0.01 0.05 0.1
This resulted in 250 different combinations. 
The value that resulted in the highest F1 value was 28 hidden nodes, 300 epochs and a learning rate of 0.1. 
    This resulted in an F1 value of 0.867.
    (Another resonable variable input was 10 hidden nodes, 500 epochs, 0.1 learning rate resulting in an F1 value of 0.866 although due to the high epochs, computation would be longer)

Untrained Neural Network: concrete28.init 
Trained network: concrete.trained               (300 epochs, 0.1 learning rate)
Results file: concrete.results

Initial weights were generated randomly in excel and then copied over to a .txt file and formatted properly. Please see the third sheet in the excel file for the init file generations. 10 files were generated with different hidden layer sizes (4 7 10 13 16 19 22 25 28 31).

Outputs were modified to booleans as specified above
The inputs were modified to be a ratio of that test cases amounts of each material in kg to the maximum amount of each material used in kg in all test cases.

Dataset came from laboratory results (found just by searching kaggle.com)
