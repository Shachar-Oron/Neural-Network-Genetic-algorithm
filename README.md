# Neural-Network-Genetic-algorithm
The program builds neural networks using a genetic algorithm that will learn patterns of files given to it and will be able to predict whether a certain string matches the pattern or not.
## Requirements

The program requires:

- Python 3.x 
- NumPy library
  
## Installation
Clone the repository:
```shell
 git clone https://github.com/Shachar-Oron/Neural-Network-Genetic-algorithm
 cd Neural-Network-Genetic-algorithm
```

Install the required dependencies:
```shell
pip install numpy
```
## Usage
Prepare the dataset:

Ensure that the dataset files are in the correct format and located in the appropriate directory.
Modify the dataset loading code in the main() function of buildnet.py to suit your dataset.
Configure the neural network and genetic algorithm parameters:

Adjust the parameters in the main() function of buildnet.py to customize the neural network architecture, genetic algorithm settings, and other relevant parameters.
Run the first the program:
```shell
python buildnet0.py
```
Run the second the program:
```shell
python buildnet1.py
```
## Monitor the progress and results:
The program will display information about the evolution process, including the fitness scores and best individuals of each generation.
The final trained neural network will be saved to a file (best_network.pickle by default).
## Features
Genetic algorithm-based training: The program utilizes a genetic algorithm to evolve neural networks and optimize their performance for pattern recognition tasks.
Customizable architecture: You can easily modify the neural network architecture by adjusting the layers, activation functions, and other parameters in the code.
Pattern recognition: The neural network is trained to recognize specific patterns in the input dataset.
