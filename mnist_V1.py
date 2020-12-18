#Library imports
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
import streamlit as st


st.title("Welcome to Grapevine Analytics")
st.markdown("Creating a neural network from scratch")
st.sidebar.title("Please select desired inouts")

opti_dict= {"Adam": keras.optimizers.Adam, "RMSprop": keras.optimizers.RMSprop, "Adagrad":keras.optimizers.Adagrad, "Adamax": keras.optimizers.Adamax, "Nadam": keras.optimizers.Nadam }
init_dict ={"RandomNormal":keras.initializers.RandomNormal(mean=0., stddev=1.),"RandomUniform" :keras.initializers.RandomUniform(),"GlorotNormal" : keras.initializers.GlorotNormal(), "GlorotUniform" :keras.initializers.GlorotUniform()}

lr_opt = st.sidebar.selectbox('Select Learning Rate',(0.1, 0.01, 1, 0.5, 0.05))
epoch_opt = st.sidebar.selectbox('Select Epochs',(20,100,200,300,400,500))
batsize_opt = st.sidebar.selectbox('Select Batch Size',(8, 16, 32, 64, 128, 256, 512, 1024, 2096))
num_layers_opt = st.sidebar.selectbox('Select Number of Hidden Layers',(2,3,4,5))
act_last_opt = st.sidebar.selectbox('Select Activation function for last layer', ("relu", "sigmoid", "tanh", "softmax"))
opti_opt = st.sidebar.selectbox('Select Optimizer', ("Adam", "RMSprop", "Adagrad", "Adamax", "Nadam"))
init_opt = st.sidebar.selectbox('Select Initializer', ("RandomNormal", "RandomUniform", "GlorotNormal", "GlorotUniform"))
regu = st.sidebar.selectbox('Wanna add regularization?', ("Yes", 'No'))
if regu == "Yes":
    regu_type =  st.sidebar.selectbox('Select regularization type?', ('l1', 'l2' ))
    regu_rate = st.sidebar.selectbox('Select regularization rate?', (0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1, 5 ))
drp =  st.sidebar.selectbox('Wanna add dropout?', ("Yes", 'No'))
if drp == "Yes":
    drp_rate =  st.sidebar.selectbox('Select dropout rate?', (0.05, 0.1, 0.15, 0.20, 0.25, 0.30,0.5))
HyperParam1 = {
        #Training Level Hyper Params
        "epochs": epoch_opt,
        "lr": lr_opt,
        "batchSize": batsize_opt,
        "opt": opti_dict.get(opti_opt),
        #Network Level HyperParam
        "NumHiddenLayers": num_layers_opt,
        1:[64, "sigmoid", init_dict.get(init_opt)],
        2:[32, "relu", init_dict.get(init_opt)],
        3:[16, "relu", init_dict.get(init_opt)],
        4:[8, "relu", init_dict.get(init_opt)],
        5:[4, "relu", init_dict.get(init_opt)]
    }




class nn_generic:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        st.write("Training label shape: ", self.y_train.shape)
        st.write("Training Data Shape", self.x_train.shape)
        self.num_classes = 10
        self.image_size = 28 * 28

    def encoding(self):
        #Setup train and test splits
        st.write("Data before one-hot encoding", self.y_train[:5])
        # Convert to "one-hot" vectors using the to_categorical function
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        st.write("Data after one-hot encoding\n", self.y_train[:5])

    def flatten_image(self):
        # Flatten the images
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.image_size)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.image_size)

    def MakeGenericModel (self, hyperP):
        try:
            self.modelH = Sequential()
            numHiddenLayers = hyperP["NumHiddenLayers"]
            act = hyperP[1][1]
            init = hyperP[1][2]

            self.modelH.add(Dense(hyperP[1][0], activation=hyperP[1][1], input_shape=(self.image_size,),#))
                                    kernel_initializer= init))

            for i in range(2, numHiddenLayers+1):
                l = hyperP[i]
                init = l[2]
                if regu == "Yes":
                    if regu_type=='l1':
                        self.modelH.add(Dense(l[0], activation=l[1], kernel_initializer=init, kernel_regularizer=keras.regularizers.l1(l1=regu_rate)))
                    else:
                        self.modelH.add(Dense(l[0], activation=l[1], kernel_initializer=init, kernel_regularizer=keras.regularizers.l2(l2=regu_rate)))
                else:
                    self.modelH.add(Dense(l[0], activation=l[1], kernel_initializer=init))


                if drp== "Yes":
                    self.modelH.add(Dropout(drp_rate))
            self.modelH.add(Dense(self.num_classes, activation=act_last_opt))

            opt =hyperP["opt"](hyperP["lr"])
            self.modelH.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
            print ("Model Created")
            self.modelH.summary()
            return self.modelH
        except Exception as ke:
            print ("Key is not defined", ke)
            self.modelH = None

    def fit_model(self, modelH):
        class LossAndErrorPrintingCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                prog.progress(epoch/epoch_opt)

        prog = st.progress(0)
        self.history = modelH.fit(self.x_train, self.y_train, batch_size=HyperParam1['batchSize'], epochs=HyperParam1['epochs'], verbose=2, validation_split=.2, callbacks=[LossAndErrorPrintingCallback()])
        prog.progress(100)

    def accuracy(self):
        loss, accuracy  = self.modelH.evaluate(self.x_test, self.y_test, verbose=False)
        loss_train, accuracy_train = self.modelH.evaluate(self.x_train, self.y_train, verbose=False)
        st.write(f'Test loss: {loss:.3}')
        st.write(f'Test accuracy: {accuracy:.3}')
        st.write(f'Train loss: {loss_train:.3}')
        st.write(f'Train accuracy: {accuracy_train:.3}')

    def plot_accuracy_graph(self):
        fig, ax = plt.subplots(1,1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()
        st.pyplot(fig)

submit = st.sidebar.button('Predict')
if submit:
    dr = nn_generic()
    dr.encoding()
    dr.flatten_image()
    modelH = dr.MakeGenericModel(HyperParam1)
    dr.fit_model(modelH)
    dr.accuracy()
    dr.plot_accuracy_graph()
