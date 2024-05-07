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

- Batch Normalization
- Dropout
- Data Augmentation
- other methods or tricks

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

- A CNN model with MaxPooling layers from Problem 2.1 in this assignment. 
- An MLP model with ReLU layers (you can modify as you want).

In your report,
```
• attach a plot of validation accuracy curves from the two models where the x axis and y axis are the number of training epochs (which is 30) and accuracy, respectively,
• analyze the results (e.g., why does one model perform better than the other?) within one paragraph.
```


### Results
#### 1.1 Convolution and Average Pooling using NumPy
As same reason, results of Filter 2 (takes other values on vertical line) and Filter 3(takes other values on other diagonal) look spread in vertical and reverse diagonal direction respectively.
In case of Average Pooling, it makes 4 parameters to 1 parameter by averaging them. That means the image information is compressed and also lost. That is reason why the results look more blurry.

<center><img src="/Figure/visualization1.png" width="50%" height="50%"></center>

<center><img src="/Figure/visualization2.png" width="50%" height="50%"></center>

  
#### 1.2 Convolution and Average Pooling using PyTorch
I define the error between two kinds of methods as euclidean distance between two results. The euclidean distance between numpy output and pytorch output by each example and each filter can be shown below. we can know that both are almost same.

<center><img src="/Figure/visualization3.png" width="50%" height="50%"></center>

<center><img src="/Figure/distance.png" width="50%" height="50%"></center>

#### 2.1 A CNN with MaxPooling layers
Let the input image size as (1, 28, 28).
- (Layer 1) Conv2d-1 (32, 3, 3) → 32 * 3 * 3 + 32 = 320 parameters
- (Layer 3) Conv2d-2 (64, 3, 3) → 64 * 32 * 3 * 3 + 64 = 18,496 parameters
- (Layer 6) Linear-1 (64, 30976) → 64 * 30976 + 64 = 1,982,528 parameters
- (Layer 7) Linear-2 (32, 64) → 32 * 64 + 32 = 2,080 parameters
- (Layer 8) Linear-3 (10, 32) → 10 * 32 + 10 = 330 parameters
Number of total parameters is 2,003,754.

<center><img src="/Figure/training_history.png" width="50%" height="50%"></center>

#### 2.2 Prevention of Overfitting

<center><img src="/Figure/training_history2.png" width="50%" height="50%"></center>

I used two methods(weight decay and dropout) to prevent overfitting. As we can see from training-validation accuracies and loss graph, the gap of accuracies and loss between training and validation with two methods is less than the gap without them. Also, the accuracy on the test images is quite increased.

The below explains why each two methods can prevent overfitting.
- Weight decay(1e-5)
  - By using Weight decay to set the lower bound of loss value, parameters can be prevented from being excessively      updated (i.e., overfitting).

- Drop out(p = 0.7)
  - Dropout prevents the corresponding parameter from being updated at that point by making certain parameters zero with the probability of p. That is, the number of times parameters are updated is reduced in probability, preventing overfitting.

#### 3. Comparison of MLP and CNN

<center><img src="/Figure/comparison.png" width="50%" height="50%"></center>

For reasonable comparison, I tried to make numbers of total trainable parameters of both model same. The CNN model has 2,003,754 parameters and the MLP model has 2,005,610 parameters. Although the number of parameters are similar, the accuracy on test images of the CNN and MLP are 91.48% and 80.54%, respectively. I think the reason why the accuracy gap exists is affect of locality. The CNN model can consider locality information of input images. In the other hands, the MLP model can’t refer their locality information because they make input images flatten immediately when they come in the model. That is reason why CNN is powerful than MLP in image task.

# ETC
For educational purpose only. This software cannot be used for any re-distribution with or without modification. The lecture notebook files are copied or modified from the material of Siamak Ravanbakhsh. 

