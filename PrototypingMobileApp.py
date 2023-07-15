from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Machine Learning-Based Prototyping of Graphical User Interfaces for Mobile Apps")
main.geometry("1300x900")

global cnn_model, filename, accuracy, precision, recall, fscore
global X_train, X_test, y_train, y_test, X, Y, labels

def uploadDataset(): 
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def preprocess():
    global filename, X, Y, labels
    text.delete('1.0', END)
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
        labels = np.load("model/labels.npy")
    else:
        X = []
        Y = []
        labels = []
        index = 0
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    code_name = os.path.splitext(directory[j])[0]
                    if os.path.exists("Dataset/Code/"+code_name+".json"):
                        img = cv2.imread(root+"/"+directory[j])
                        img = cv2.resize(img, (64,64))
                        im2arr = np.array(img)
                        im2arr = im2arr.reshape(64,64,3)
                        for k in range(0,5):
                            X.append(im2arr)
                            Y.append(index)
                            labels.append(code_name)
                        print(directory[j]+" "+code_name+" "+str(index))
                        index = index + 1
        X = np.asarray(X)
        Y = np.asarray(Y)
        labels = np.asarray(labels)
        np.save("model/X", X)
        np.save("model/Y", Y)
        np.save("model/labels", labels)
    text.insert(END,"Image processing task completed\n\n")
    text.insert(END,"Total Android GUI Components found in dataset : "+str(X.shape[0])+"\n")
    text.update_idletasks()
    test = X[1003]
    cv2.imshow("Processed Image",cv2.resize(test,(150,150)))
    cv2.waitKey(0)

def splitDataset():
    global filename, X, Y, labels
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% images used to train CNN algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% images used to test CNN algorithms : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    
def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()
    label = np.unique(labels)
    label = label[0:30]
    y_test = y_test[0:30]
    predict = predict[0:30]
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = label, yticklabels = label, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(label)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runCNN():
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, cnn_model
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn_model = model_from_json(loaded_model_json)
        json_file.close()
        cnn_model.load_weights("model/model_weights.h5")
        cnn_model._make_predict_function()       
    else:
        cnn_model = Sequential()
        cnn_model.add(Convolution2D(32, 3, 3, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
        cnn_model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(output_dim = 256, activation = 'relu'))
        cnn_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
        cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = cnn_model.fit(X_train, y_train, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data=(X_test, y_test))
        cnn_model.save_weights('model/model_weights.h5')            
        model_json = cnn_model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)        
    calculateMetrics("CNN", predict, y_test)

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Error Rate')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['CNN Accuracy', 'CNN Loss'], loc='upper left')
    plt.title('CNN Accuracy & Loss Graph')
    plt.show()

def predict():
    text.delete('1.0', END)
    global cnn_model
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = cnn_model.predict(img)
    predict = np.argmax(preds)
    print(np.amax(preds))
    name = 'Dataset/Data/'+labels[predict]+".json"
    if os.path.exists(name) and np.amax(preds) > 0.90:
        with open(name, "rb") as file:
            data = file.read()
        file.close()
        text.insert(END,data.decode())
        with open("code.json", "wb") as file:
            file.write(data)
        file.close()
    else:
        text.insert(END,"Unable to generate code fro give GUI")

font = ('times', 16, 'bold')
title = Label(main, text='Machine Learning-Based Prototyping of Graphical User Interfaces for Mobile Apps',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)

uploadButton = Button(main, text="Upload RICO Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocess, bg='#ffb3fe')
processButton.place(x=310,y=550)
processButton.config(font=font1) 

splitButton1 = Button(main, text="Shuffling, Splitting & Dataset Normalization", command=splitDataset, bg='#ffb3fe')
splitButton1.place(x=570,y=550)
splitButton1.config(font=font1) 

cnnButton = Button(main, text="Run CNN Algorithm", command=runCNN, bg='#ffb3fe')
cnnButton.place(x=50,y=600)
cnnButton.config(font=font1) 

graphButton = Button(main, text="CNN Training Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=310,y=600)
graphButton.config(font=font1) 

predictButton = Button(main, text="Predict Code from Image", command=predict, bg='#ffb3fe')
predictButton.place(x=570,y=600)
predictButton.config(font=font1) 

main.config(bg='chocolate1')
main.mainloop()
