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

#### 1.1. Convolution and Average Pooling using NumPy
Implement two simple forward networks as follows:

<img src="/Figure/CNN_architecture.png" width="50%" height="50%">

In your report,
```
• attach the visualization results and
• write down your analysis stating the difference between results (Max. 600 characters). 
Note that you must obtain 6 different visualizations.
```

#### 1.2. Convolution and Average Pooling using PyTorch
Implement the above CNN models using PyTorch.

In your report,
```
• attach the visualization results and
• provide whether your implementation using NumPy is the same as that using PyTorch by cal- culating any errors.
```

### Problem 2: Convolutional Neural Networks (CNN)
In this part, implement a convolutional neural network (CNN) on the <b>FashionMNIST</b> dataset for an image classification task. You can now use the [PyTorch](https://pytorch.org/) library for network design and construction. You have to fill your code in the blank section following the “PLACE YOUR CODE HERE” comments in the CNN_problem_2.ipynb file.

#### 2.1. A CNN with MaxPooling layers
You must implement a CNN model under the CNN Max() class. The model has a sequential structure:

<img src="/Figure/CNN_structure.png" width="50%" height="50%">

where the convolution layer is with bias=True, which needs to be accounted for calculating the number of parameters.
All other arguments use default values. You will also implement forward and backward passes to optimize CNN by using stochastic gradient descent (SGD) with a momentum method. Note that your test accuracy should be over 90.2% on the test images.

In your report,
```
• analyze the number of parameters used in each layer and the total number of parameters over the entire model considering the input image size,
• attach the graph of training and validation accuracies over 30 epochs,
• attach the capture of the test accuracy on the 10, 000 test images from the ipynb screen.
```

#### 2.2. Prevention of Overfitting
<b>Overfitting</b> happens when your model fits too well to the training set. It then becomes difficult for the model to generalize to new examples that were not in the training set. For example, your model recognizes specific images in your training set instead of general patterns. Your training accuracy will be higher than the accuracy on the validation/test set. To reduce overfitting, you must implement techniques such as followings (but are not limited to) to the CNN model in Problem 2.1:

• Batch Normalization
• Dropout
• Data Augmentation
• other methods or tricks

You can decide to set the parameters for the regularizations (e.g. dropout rate).

In your report,
```
• attach a graph of training-and-validation accuracies over 30 epochs with a selected technique that handles overfitting,
• report the test accuracy on the 10, 000 test images,
• compare the results of the CNN model without regularization methods (analyze and explain why the selected technique works for preventing overfitting).
```

### Problem 3: Comparison of MLP and CNN
In this problem, you will compare a CNN model with MLP and analyze the performance difference by computing validation accuracies given the Fashion MNIST image dataset. You have to modify the code in the CNN_problem_3.ipynb file.

Compare the validation accuracies of two models:

• A CNN model with MaxPooling layers from Problem 2.1 in this assignment. 
• An MLP model with ReLU layers (you can modify as you want).

In your report,
```
• attach a plot of validation accuracy curves from the two models where the x axis and y axis are the number of training epochs (which is 30) and accuracy, respectively,
• analyze the results (e.g., why does one model perform better than the other?) within one paragraph.
```


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

