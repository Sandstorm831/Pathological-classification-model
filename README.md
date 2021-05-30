
# Pathological-classification-model
A classifier used to classify different OCT images as CME, DMV , DRUSEN &amp; NORMAL
In this project I have made a model , that would predict whether a OCT scan belong to DME, CNV, DRUSSEN or NORMAL class 
# Importing libraries
```
import numpy as np
import matplotlib.pyplot as plt
import os

from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers

from tensorflow.keras.applications import resnet50
```
In my project, libraries I have used :
- Numpy
- Matplotlib
- Os
- IPython
- Tensorflow
- Scikit Learn
 
# Extracting classes
```
main_folder = 'OCT2017/train'
class_names = os.listdir(main_folder)
print(class_names)
```
Here we use directories for finding the different classes 
of the data . For listing the directories , we use OS 
library’s listdir method .


# Loading dataset
```
x_train = [] #store the array of train images
y_train = [] #store the arrays labels

for folder in os.listdir(main_folder):
    image_list = os.listdir(main_folder+'/'+folder)
    for img_name in image_list:
        img = image.load_img(main_folder+'/'+folder+'/'+img_name, target_size=(100,100))
        # converting images to arrays of rgb values
        img=image.img_to_array(img)
        
        # to preprocess images before passing to resnet
        img = resnet50.preprocess_input(img)
        
        # adding image matrix to the input image matrix - x
        x_train.append(img)
        y_train.append(class_names.index(folder)) # adding label to the label matrix - y
```

Two for loop are taken into account for the iterating over all examples of the image, first for loop iterate over all the classes of the data and second for loop is used to iterate over all the images of that particular class 
Converts the image in a 100 x 100 pixels image  
Converts the 100 x 100 image into an array of dimension (100,100,3)
Appending the matrix to a new matrix x_train
Appending the class of the image to the y_train matrix
Similar procedure is followed in creating the test dataset 

# Converting Arrays
```
x_test = np.array(x_test) # converting x_test to numpy array
y_test = to_categorical(y_test) # one hot encoding for the labels
```
x_train & x_test is an array so , converting them to numpy 
arrays 
applying one hot encoding to the labels of the classes

# Splitting the data into training and validation data 
```
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=0)
``` 
90% of the data is used for the training the data while 10% of the data is set as validation data 

# ResNet-50 CNN Classifier
ResNet-50 is a convolutional neural network that is 50 layers deep . In this project , pretrained model of resnet 50 is used. 
```
model_resnet = resnet50.ResNet50(weights='imagenet')
model_resnet.summary()
```
![image](https://user-images.githubusercontent.com/76916164/120118825-360b9400-c1b2-11eb-93d1-3dcde5fb20b5.png)


Retaining all the parameters of the Resnet-50
Printing a general ResNet-50 classifier summary

# Changing the dimensions of input layer

As the dimensions of the input layer is different from dimensions of the input image matrix , therefore , add a new input layer with dimensions corresponding to the input matix
```
input_layer = layers.Input(shape=(100,100,3))
model_resnet = resnet50.ResNet50(weights='imagenet',input_tensor=input_layer,include_top=False)
```

Initializing a input layer of 100 x 100 x 3 dimension
Passing the initialized input layer to the ResNey-50 model

# Flattening and adding output layers
```
last_layer = model_resnet.output
flatten = layers.Flatten()(last_layer)# last layer is flattened
output_layer = layers.Dense(4,activation = 'softmax')(flatten) # creating flattened output dense layer 
model = models.Model(inputs = input_layer, outputs = output_layer)
```
 
Flattening the last layer 
Adding a flattened dense output layer with 4 output channels with softmax activation layer 
# Freezing all the parameters except last layer
```
for layer in model.layers[:-1]:
    layer.trainable=False
    
# final model structure
model.summary()
```
![image](https://user-images.githubusercontent.com/76916164/120118964-ce097d80-c1b2-11eb-8c43-95fe17f14991.png)
For loop iterates over all layers except last layer , freezing all the parameters except of last layer
![image](https://user-images.githubusercontent.com/76916164/120119019-0c9f3800-c1b3-11eb-93fb-dd21b5c9a072.png)

In model summary we can see that input layer have dimensions as that of the image matrix
 
Here , only 131076 parameters are trainable out of 23718788 parameters , therefore we have successfully freezed all the parameters except that of last layer
# Compiling and Training 
```
# compiling the model 
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
```
```
# fitting the data into model
model.fit(X_train,Y_train,epochs=5,batch_size=64,verbose=True,validation_data=(X_val,Y_val))
```
![image](https://user-images.githubusercontent.com/76916164/120119078-58ea7800-c1b3-11eb-80c6-719534163c7f.png)

Final training accuracy  = 86.39%           while         validation accuracy = 80.63%
# Accuracy and F1 scores 
```
def predict(img_name):
    prediction = model.predict(img_name.reshape(1,100,100,3))
    return(np.argmax(prediction))
```
```
output = []
for image_no in range(x_test.shape[0]) :
    output.append(predict(x_test[image_no]))
```
```
output = np.array(output).reshape(1000,1)
```
```
test_folder = 'OCT2017/test'
true_labels = []
for label in os.listdir(test_folder):
    for img_number in os.listdir(test_folder+'/'+label):
        true_labels.append(class_names.index(label))     
```
```
from sklearn.metrics import f1_score,accuracy_score
f1_score(true_labels,output, average=None)
```
![image](https://user-images.githubusercontent.com/76916164/120119162-d0200c00-c1b3-11eb-89bd-414411ecd53d.png)
```
output = to_categorical(output)
```
```
from tensorflow.keras.metrics import Accuracy
```
```
acc = Accuracy()
acc.update_state(y_test,output)
acc.result().numpy()
```
![image](https://user-images.githubusercontent.com/76916164/120119224-2725e100-c1b4-11eb-93b7-177c1381ac4c.png)

Due to class imbalance the F1 score of different of classes differ so drastically . different class have different numbers of data thus model have been trained on different amount of data for different class . 

At last I have visualized the region of interest (ROI) of the images to identify which portion of the image is particularly responsible for the allocation of particular class


 ![image](https://user-images.githubusercontent.com/76916164/120119254-550b2580-c1b4-11eb-9a8e-7a0a5bcaee4c.png)
 
 
Here the yellow region shows the portion of image which is responsible for the 
classification of image 

They have been formed by the superimposition of heatmap of the images over 
the images


references :

Dataset: https://data.mendeley.com/datasets/rscbjbr9sj/2

[1] Pytorch - https://pytorch.org/tutorials
 

[2] Deep Learning for AI Specialization
https://www.deeplearning.ai/program/ai-for-medicine-specialization/
 

[3] Roychowdhury, Sohini, et al. “SISE-PC: Semi-supervised Image Subsampling for Explainable Pathology.” arXiv preprint arXiv:2102.11560 (2021). https://github.com/anoopsanka/retinal_oct  
 
[4] Feature Visualization: GradCAM and torchcam
GradCAM: https://keras.io/examples/vision/grad_cam (tensorflow)
torchcam: https://pypi.org/project/torchcam (pytorch)

