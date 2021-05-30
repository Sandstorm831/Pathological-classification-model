
# Pathological-classification-model
A classifier used to classify different OCT images as CME, DMV , DRUSEN &amp; NORMAL
In this project I have made a model , that would predict whether a OCT scan belong to DME, CNV, DRUSSEN or NORMAL class 
# Importing libraries
![image](https://user-images.githubusercontent.com/76916164/120117734-402a9400-c1ac-11eb-89f7-c836140dc379.png)   

# Extracting classes
 ![image](https://user-images.githubusercontent.com/76916164/120117755-53d5fa80-c1ac-11eb-93c5-56153d60bca4.png)

# Loading dataset
![image](https://user-images.githubusercontent.com/76916164/120117770-6819f780-c1ac-11eb-9d59-e27ee7e265bb.png)

Two for loop are taken into account for the iterating over all examples of the image, first for loop iterate over all the classes of the data and second for loop is used to iterate over all the images of that particular class 
Converts the image in a 100 x 100 pixels image  
Converts the 100 x 100 image into an array of dimension (100,100,3)
Appending the matrix to a new matrix x_train
Appending the class of the image to the y_train matrix
Similar procedure is followed in creating the test dataset 

# Converting Arrays
 ![image](https://user-images.githubusercontent.com/76916164/120117790-85e75c80-c1ac-11eb-98b6-5763e7b01c02.png)

# Splitting the data into training and validation data 
![image](https://user-images.githubusercontent.com/76916164/120117807-a4e5ee80-c1ac-11eb-974d-df85292906e1.png)
 
 
90% of the data is used for the training the data while 10% of the data is set as validation data 

# ResNet-50 CNN Classifier
ResNet-50 is a convolutional neural network that is 50 layers deep . In this project , pretrained model of resnet 50 is used. 
![image](https://user-images.githubusercontent.com/76916164/120117821-bc24dc00-c1ac-11eb-8228-84a0569ac33b.png)

Retaining all the parameters of the Resnet-50
Printing a general ResNet-50 classifier summary
# Changing the dimensions of input layer
As the dimensions of the input layer is different from dimensions of the input image matrix , therefore , add a new input layer with dimensions corresponding to the input matix
 ![image](https://user-images.githubusercontent.com/76916164/120117834-d5c62380-c1ac-11eb-9bb7-3cfe38f80038.png)

Initializing a input layer of 100 x 100 x 3 dimension
Passing the initialized input layer to the ResNey-50 model

# Flattening and adding output layers
![image](https://user-images.githubusercontent.com/76916164/120117883-0f972a00-c1ad-11eb-8dc1-f88c5a9e69ee.png)
 
Flattening the last layer 
Adding a flattened dense output layer with 4 output channels with softmax activation layer 
# Freezing all the parameters except last layer
![image](https://user-images.githubusercontent.com/76916164/120117896-2f2e5280-c1ad-11eb-8371-83e20866c8c3.png)

For loop iterates over all layers except last layer , freezing all the parameters except of last layer
In model summary we can see that input layer have dimensions as that of the image matrix
 
Here , only 131076 parameters are trainable out of 23718788 parameters , therefore we have successfully freezed all the parameters except that of last layer
# Compiling and Training 
![image](https://user-images.githubusercontent.com/76916164/120117920-52590200-c1ad-11eb-9b25-eba80d0dd8de.png)

Final training accuracy  = 86.39%           while         validation accuracy = 80.63%
# Accuracy and F1 scores 
![image](https://user-images.githubusercontent.com/76916164/120117939-6f8dd080-c1ad-11eb-887b-8c290048c423.png)

Due to class imbalance the F1 score of different of classes differ so drastically . different class have different numbers of data thus model have been trained on different amount of data for different class . 

At last I have visualized the region of interest (ROI) of the images to identify which portion of the image is particularly responsible for the allocation of particular class
 ![image](https://user-images.githubusercontent.com/76916164/120117973-ab289a80-c1ad-11eb-9850-955524ea17f5.png)


references :

Dataset: https://data.mendeley.com/datasets/rscbjbr9sj/2

[1] Pytorch - https://pytorch.org/tutorials
 

[2] Deep Learning for AI Specialization
https://www.deeplearning.ai/program/ai-for-medicine-specialization/
 

[3] Roychowdhury, Sohini, et al. “SISE-PC: Semi-supervised Image Subsampling for Explainable Pathology.” arXiv preprint arXiv:2102.11560 (2021). https://github.com/anoopsanka/retinal_oct  
 
[4] Feature Visualization: GradCAM and torchcam
GradCAM: https://keras.io/examples/vision/grad_cam (tensorflow)
torchcam: https://pypi.org/project/torchcam (pytorch)

