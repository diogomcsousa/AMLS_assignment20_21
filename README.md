# AMLS_assignment_20_21

This work contains four tasks in the computer vision area, more specifically image processing with the defined objective
for a better understanding of machine learning systems and acquire experience in dealing with real-world data. 
Two of the proposed tasks are binary classiﬁcation problems, on Gender Detection and Smile Detection. The other two 
tasks are multiclass classification problems on Face Shape Detection and Eye Color Detection. The pipeline used 
to solve this tasks consists of three steps. First it is done a feature extraction from the region of interest (ROI), 
then the model is selected and trained with different values for its parameters, finally a test set is used to get 
performance of the model. For these tasks different techniques for feature extraction were used, such as Face Encoding, 
Facial Landmark Detector and Haar Cascade features. Regarding the model, it was used an SVM polynomial kernel 
classifier for every task, with different parameters. After this process different performances were verified, from 72% to 98%.

### Installation
Run the following commands to install the needed python libraries prior to run the code
```
$ pip3 install dlib
$ pip3 install numpy
$ pip3 install keras
$ pip3 install opencv_python
$ pip3 install sklearn
$ pip3 install face_recognition
$ pip3 install matplotlib
$ pip3 install tensorflow
```

### How to run the code
Run the main.py file to execute all four tasks, on root directory
```
$ python3 main.py
```


### Project Organisation
 ```
 .
 |-- main.py                      # Main file that will run the four tasks and call other Python files
 |-- A1/
 |   |__ a1_task.py               # Class Gender Detection that will use classifiers and data_preprocessing files to perform feature extraction, train and test the model
 |-- A2/
 |   |__ a2_task.py               # Class Smile Detection that will use classifiers and data_preprocessing files to perform feature extraction, train and test the model
 |-- B1/
 |   |__ b1_task.py               # Class Face Shape Detection that will use classifiers and data_preprocessing files to perform feature extraction, train and test the model
 |-- B2/
 |   |__ b2_task.py               # Class Eye Color Detection that will use classifiers and data_preprocessing files to perform feature extraction, train and test the model
 |-- Datasets/                    # Folder containing the datasets used on four tasks
 |-- Utils/
 |   |__ classifiers.py           # Contains classifiers used in training the models
 |   |__ data_analysis.py         # Create learning curve plots
 |___|__ data_preprocessing.py    # Consists of functions used in different phases of feature engineering
 ```
