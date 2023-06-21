# Neural-Network-Genetic-algorithm
The program builds neural networks using a genetic algorithm that will learn patterns of files given to it and will be able to predict whether a certain string matches the pattern or not.
## Requirements for the Python files

The program requires:

- Python 3.x 
- NumPy library
  
## Installation
Clone the repository:
```shell
 git clone https://github.com/Shachar-Oron/Neural-Network-Genetic-algorithm
 cd Neural-Network-Genetic-algorithm
```

If you want to run the python files, install the required dependencies:
```shell
pip install numpy
```
## Program Structure
The buildnet program is divided into 4 parts:
- Reading the data given as input from the user and placing it as the relevant variables.
- The Individual class representing a network of neurons, an individual in the population. The class contains the forward and prediction functions required for forward propagation.
- The class of the genetic algorithm that tries to find the network with the weights that yield the best result using the fitness function.
- Running the predict and forward function on the test set with the neural network with the highest fitness that we got from the genetic algorithm on the training set.
- Creating a went file containing the structure of the best neural network provided by the genetic algorithm.

The runnet program is divided into 2 parts:
- Reading the test file that the user provides and reading the wnet file
- Running the predict function with the network we received from wnet on the inputs received from the test file.
- Creating a results file containing the calculated labels for each input in the test file.

## Prepare the dataset:

Ensure that the dataset files are in the correct format and located in the appropriate directory.
Modify the dataset loading code in the main() function of buildnet.py to suit your dataset.

Configure the neural network and genetic algorithm parameters:

Adjust the parameters in the main() function of buildnet.py to customize the neural network architecture, genetic algorithm settings, and other relevant parameters.

## Run the program:
Run the first the program:
```shell
 ./buildnet0.exe
```
Then the app will ask you to input your test and train files for buildnet0
Run the second the program:
```shell
 ./buildnet1.exe
```
Then the app will ask you to input your test and train files for buildnet1
Both apps will create ‘wnet.pkl’ files as outputs that hold the best neural network structure for each runnet .exe files. 
To run the ‘runnet0.exe’ program, enter the following command:
```shell
./runnet0.exe
```
Now do the same for runnet1.exe with the relevant input file:
```shell
./runnet1.exe
```
The final results will be in the files: results0.txt & results1.txt

## Monitor the progress and results:
The program will display information about the evolution process, including the fitness scores and best individuals of each generation.
The final trained neural network will be saved to a file (wnet0.pickle & wnet1.pickle by default).
The program will also display the results of the neural network on the test set after you will run the runnet0.exe and runnet1.exe programs.

## Features
Genetic algorithm-based training: The program utilizes a genetic algorithm to evolve neural networks and optimize their performance for pattern recognition tasks.
Customizable architecture: You can easily modify the neural network architecture by adjusting the layers, activation functions, and other parameters in the code.
Pattern recognition: The neural network is trained to recognize specific patterns in the input dataset.
