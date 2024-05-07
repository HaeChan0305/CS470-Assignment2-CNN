<div align=center>
  <h1>
    CNN: Convolutional Neural Networks
  </h1>
  <p>
    <b>KAIST CS470: Introduction to Artificial Intelligence (Spring 2023)</b><br>
    Programming Assignment 2
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://sites.google.com/site/daehyungpark target="_blank"><b>Daehyung Park</b></a> (daehyung [at] kaist.ac.kr)<br>
  </p>
</div>


## Assignment 2
### Problem 1: Convolution operations using NumPy
In this problem, you will implement convolution and average pooling functions using [NumPy](https://numpy.org/). You will then analyze the result on the MNIST dataset for a handwriting digit classification task. You will also compare the result with that using [Pytorch](https://pytorch.org/). You can now use the PyTorch library for network design and construction.
You have to fill your code in the blank section following the “PLACE YOUR CODE HERE” comments in the <b>CNN_problem_1.ipynb</b> file.

#### 1.1 Convolution and Average Pooling using NumPy
Implement two simple forward networks as follows:

<img src="/Figure/CNN_architecture.png" width="50%" height="50%">

#### 1.2 Convolution and Average Pooling using PyTorch
Implement the above CNN models using PyTorch.


### Problem 2: Convolutional Neural Networks (CNN)
In this part, implement a convolutional neural network (CNN) on the <b>FashionMNIST</b> dataset for an image classification task. You can now use the [PyTorch](https://pytorch.org/) library for network design and construction. You have to fill your code in the blank section following the “PLACE YOUR CODE HERE” comments in the CNN_problem_2.ipynb file.

### TASK 3: Visualization
You have to plot the loss function and the accuracies on the training and validation sets. Then, visualize the weights that were learned in the first layer of the network. The weights of the intermediate layer may learn to represent specific features of the inputs, such as their curvature, thickness, or orientation.

### Results
- Visualization
  
  <center><img src="/Figure/visualization.png" width="50%" height="50%"></center>
  
- Loss and accuracy Plot
  - ReLU

    <center><img src="/Figure/loss_ReLU.png" width="50%" height="50%"></center>
    
    <center><img src="/Figure/accuracy_ReLU.png" width="50%" height="50%"></center>

  - Leaky ReLU

    <center><img src="/Figure/Leaky_ReLU.png" width="50%" height="50%"></center>

  - SWISH

    <center><img src="/Figure/SWISH.png" width="50%" height="50%"></center>

  - SELU

    <center><img src="/Figure/SELU.png" width="50%" height="50%"></center>

- Loss and accuracy by activation functions on 2000 iteration.

| Activation Function | ReLU | Leaky ReLU | SWISH | SELU |
|---|---|---|---|---|
| **Loss** | 1.49 | 1.50 | 1.51 | 1.54 |
| **Accuracy** | 0.470  | 0.469 | 0.472 | 0.462 |


<!--
# Tutorial Links
- [Tutorial 1-1](https://github.com/pidipidi/CS470_IAI_2023_Spring/blob/main/tutorial_1/cs470_tutorial_1_1.ipynb)
- [Tutorial 1-2](https://github.com/pidipidi/CS470_IAI_2023_Spring/blob/main/tutorial_1/cs470_tutorial_1_2.ipynb)
- [Tutorial 1-3](https://github.com/pidipidi/CS470_IAI_2023_Spring/blob/main/tutorial_1/cs470_tutorial_1_3.ipynb)
- [Tutorial 2](https://github.com/pidipidi/CS470_IAI_2023_Spring/blob/main/tutorial_2/RL_tutorial.ipynb)
- [Tutorial 3](https://github.com/pidipidi/CS470_IAI_2023_Spring/blob/main/tutorial_3/README.md)


# Quiz
- [Quiz 1](https://github.com/pidipidi/CS470_IAI_2023_Spring/blob/main/tutorial_1/MLP_tutorial_quiz_problem.ipynb)
- [Quiz 2](https://github.com/pidipidi/CS470_IAI_2023_Spring/blob/main/tutorial_1/tutorial2_quiz.ipynb)


# Installation
- [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation.html)
-->

# ETC
For educational purpose only. This software cannot be used for any re-distribution with or without modification. The lecture notebook files are copied or modified from the material of Siamak Ravanbakhsh. 

