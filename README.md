# Project-2: Machine Learning Based Prototyping of Graphical User Interfaces for Mobile Apps
Abstract: 
  It is common practice for developers of user-facing software to transform a mock-up of a graphical user interface (GUI) into code. This process takes place both at an application’s inception and in an evolutionary context as GUI changes keep pace with evolving features. Unfortunately, this practice is challenging and time-consuming. In this project, we present an approach that automates this process by enabling accurate prototyping of GUIs via three tasks: detection, classification, and assembly. 
  
Introduction:
  Most modern user-facing software applications are GUI-centric and rely on attractive user interfaces (UI) and intuitive user experiences (UX) to attract customers, facilitate the effective completion of computing tasks, and engage users. Software with cumbersome or aesthetically displeasing UIs is far less likely to succeed, particularly as companies look to differentiate their applications from competitors with similar functionality.

Steps of Implementing the Project:
1. Upload the RICO Dataset into an application
2. Preprocess of Dataset to read and process image
3. Split, Shuffle, and Normalize the images
4. Application Shows the train and test images
5. Perform CNN Algorithm
6. Application shows CNN accuracy, precision, recall, FScore rate result
7. Application predicts and displays the Confusion Matrix Result
8. On selecting the CNN training graph Displays the comparison of the CNN Accuracy and Loss graph
9. Select Predict code from the Image which is set to upload Android GUI
10. Load the image into the Application and displays the predicted code
11. Then compare the generated code in JSON format and the predicted code which shows same formattable of an image.

Modules:

  1.	Upload RICO Dataset: using this module we will upload the dataset to the application
  2.	Preprocess Dataset: using this module we will read each image and then resize and normalize all pixel values from the image
  3.	Shuffling, Splitting & Dataset Normalization: using this module we will shuffle the dataset and then split the dataset into train and test where the application will use 80% dataset for training and 20% for testing
  4.	Run CNN Algorithm: now 80% of train data will be input to train CNN and then apply 20% of test data to calculate CNN prediction accuracy confusion matrix
  5.	CNN Training Graph: using this module we will plot CNN training accuracy and loss graph
  6.	Predict Code from Image: using this module we will upload a test GUI screen and then CNN will predict Android code in JSON format.

Tools and Technologies:

Operating system: Windows 8 or Above. 
Coding Language: Python
Libraries: Tensorflow, Numpy, Pandas, Matplotlib, Scikit-Learn 




  
