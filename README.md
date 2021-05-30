# Pathological-classification-model
A classifier used to classify different OCT images as CME, DMV , DRUSEN &amp; NORMAL
In this project I have made a model , that would predict whether a OCT scan belong to DME, CNV, DRUSSEN or NORMAL class  
 ![image](https://user-images.githubusercontent.com/76916164/120117042-d3fa6100-c1a8-11eb-854d-0e5e076f7aad.png)

Extracting classes
 
Loading dataset
C 
Two for loop are taken into account for the iterating over all examples of the image, first for loop iterate over all the classes of the data and second for loop is used to iterate over all the images of that particular class 
Converts the image in a 100 x 100 pixels image  
Converts the 100 x 100 image into an array of dimension (100,100,3)
Appending the matrix to a new matrix x_train
Appending the class of the image to the y_train matrix
Similar procedure is followed in creating the test dataset 
Converting Arrays
 
Splitting the data into training and validation data 
 
90% of the data is used for the training the data while 10% of the data is set as validation data 
ResNet-50 CNN Classifier
ResNet-50 is a convolutional neural network that is 50 layers deep . In this project , pretrained model of resnet 50 is used. 


Retaining all the parameters of the Resnet-50
Printing a general ResNet-50 classifier summary
Changing the dimensions of input layer
As the dimensions of the input layer is different from dimensions of the input image matrix , therefore , add a new input layer with dimensions corresponding to the input matix
 
Initializing a input layer of 100 x 100 x 3 dimension
Passing the initialized input layer to the ResNey-50 model
Flattening and adding output layers
 
Flattening the last layer 
Adding a flattened dense output layer with 4 output channels with softmax activation layer 
Freezing all the parameters except last layer
 
For loop iterates over all layers except last layer , freezing all the parameters except of last layer
In model summary we can see that input layer have dimensions as that of the image matrix
 
Here , only 131076 parameters are trainable out of 23718788 parameters , therefore we have successfully freezed all the parameters except that of last layer
Compiling and Training 
 
Final training accuracy  = 86.39%           while         validation accuracy = 80.63%
Accuracy and F1 scores 
 
Due to class imbalance the F1 score of different of classes differ so drastically . different class have different numbers of data thus model have been trained on different amount of data for different class . 
At last I have visualized the region of interest (ROI) of the images to identify which portion of the image is particularly responsible for the allocation of particular class
 
