# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:16:46 2022

@author: iacov
"""
# Importing the libraries
from os.path import exists
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import math
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
plt.style.use('fivethirtyeight')
# multivariate multi-step encoder-decoder lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import metrics
from keras.callbacks import ModelCheckpoint


class DNN_LSTM:
    def __init__(self, name, csv_file,qosElementNumber=6):
        self.qosElementNumber=qosElementNumber
        self.name = name
        self.csv_file = csv_file
        self.timesteps_n_steps_in=0
        self.n_features=0
        self.n_steps_out=0         
        self.model = self.load_or_create_model(name, csv_file,qosElementNumber)
        # TEST
        array = [10]
        #self.predictsSetupExecute(array,qosElementNumber)

    def load_or_create_model(self, name, csv_file,qosElementNumber):
        if exists(name+str(qosElementNumber)+'.tf'):
            model = keras.models.load_model(name+str(qosElementNumber)+'.tf')
        else:
            model = self.createModel(name, csv_file,qosElementNumber)
        return model

    def createModel(self, name, csv_file,qosElementNumber):
        train_X, train_y, val_X, val_y,  test_X, test_y = self.load_dataset(
            csv_file,qosElementNumber)
      
        print(train_X)      
        print(train_y)
        print(self.timesteps_n_steps_in)
        print(self.n_features)
         
        # design network
        # define model
        model = Sequential()
        model.add(tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(self.timesteps_n_steps_in,self.n_features), return_sequences = True))          
        model.add(LSTM(64, activation='relu', input_shape=(self.timesteps_n_steps_in,self.n_features)))
        model.add(RepeatVector(self.n_steps_out))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(self.n_features)))
        model.compile(optimizer='adam', loss='mse', metrics=["acc",metrics.mean_squared_error, metrics.mean_absolute_error])
        # fit model
        checkp = ModelCheckpoint('./new_model.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)
        history = model.fit(train_X, train_y, epochs=1000, verbose=1,validation_data=(
            val_X, val_y), callbacks = [checkp])        
        model.summary()
        

        # make predictions
        train_predict = model.predict(train_X)
        val_predict = model.predict(val_X)

        # Print error
        print(train_y[0])
        print(train_predict[0])
        vale = input("Enter your value: "+str(qosElementNumber))
        print(vale)           
   
        
        plt.figure(figsize = (7,3))
        #plt.figure(figsize = (7, 3))
        plt.subplot(2,1,1)
        plt.plot(history.history['loss'],linewidth=2)
        plt.plot(history.history['val_loss'],linewidth=1.75)
        plt.legend(['Training', 'Validation'],fontsize=17)
        plt.xlabel('Epochs',fontsize=24)
        plt.ylabel('Losses',fontsize=24)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)


        plt.subplot(2,1,2)
        plt.plot(history.history['acc'],linewidth=2)
        plt.plot(history.history['val_acc'],linewidth=1.75)
        plt.ylabel('Accuracy',fontsize=24)
        plt.xlabel('Epochs',fontsize=24)
        plt.legend(['Training', 'Validation'], loc='lower right',fontsize=17)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.show(block = False)        
        plt.show()        
        #self.print_error(train_y, val_y, train_predict, val_predict)
        #self.return_rmse(train_y, train_predict)

        #keras.models.save_model(model, name+str(qosElementNumber)+'.tf')



        plt.figure(figsize = (7,3))
        #plt.figure(figsize = (7, 3))
        plt.subplot(2,1,1)
        plt.plot(history.history['loss'],linewidth=2)
        plt.plot(history.history['val_loss'],linewidth=1.75)
        plt.legend(['Training', 'Validation'],fontsize=17)
        plt.xlabel('Epochs',fontsize=24)
        plt.ylabel('Losses',fontsize=24)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)


        plt.subplot(2,1,2)
        plt.plot(history.history['mean_squared_error'],linewidth=2)
        #plt.plot(history.history['mean_absolute_error'],linewidth=1.75)
        plt.ylabel('mse',fontsize=24)
        plt.xlabel('Epochs',fontsize=24)
        plt.legend(['mse'], loc='lower right',fontsize=17)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.show(block = False)
        # plot the actual price, prediction in test data=red line, actual price=blue line
        #plt.plot(testPredictPlot)
        plt.show()



        return model


    def temporalize(self, X, y, lookback):
        output_X = []
        output_y = []
        for i in range(len(X)-lookback-1):
            t = []
            for j in range(1,lookback+1):
                # Gather past records upto the lookback period
                t.append(X[[(i+j+1)], :])
            output_X.append(t)
            output_y.append(y[i+lookback+1])
        return output_X, output_y


    # split a multivariate sequence into samples
    def split_sequences(self,sequences, n_steps_in, n_steps_out):
    	X, y = list(), list()
    	for i in range(len(sequences)):
    		# find the end of this pattern
    		end_ix = i + n_steps_in
    		out_end_ix = end_ix + n_steps_out
    		# check if we are beyond the dataset
    		if out_end_ix > len(sequences):
    			break
    		# gather input and output parts of the pattern
    		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
    		X.append(seq_x)
    		y.append(seq_y)
    	return array(X), array(y)
     





    def load_dataset(self, csv_file,qosElementNumber):
      dataframe = pd.read_csv(csv_file)
      
      print(dataframe.head())
      train, val, test = np.split(dataframe.sample(
                frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])
      print(len(train), 'training examples', train.head())
      print(len(val), 'validation examples', val.head())
      print(len(test), 'test examples', test.head())
      

      self.timesteps_n_steps_in = 3
      self.n_steps_out=2       
      self.n_features = int(qosElementNumber)
      
    
      # diplay the contents of the csv file with NO processing
      myData_processed_train = train.iloc[:, 1:int(qosElementNumber)+1].values
      #myData_processed_train[:, 0] = myData_processed_train[:, 0].astype(int)
      for i in range(0,int(qosElementNumber)-1,1):
          myData_processed_train[:, i] = myData_processed_train[:, i].astype(float)  
          
      print(myData_processed_train)
   
      myData_processed_val = val.iloc[:,1:int(qosElementNumber)+1].values
      #myData_processed_val[:, 0] = myData_processed_val[:,0].astype(int)
      for i in range(0,int(qosElementNumber)-1,1):      
          myData_processed_val[:, i] = myData_processed_val[:,i].astype(float) 
          
      print(myData_processed_val)
      #print(val)
      #vale = input("Enter your value: "+str(qosElementNumber))
      #print(vale)    
    
      myData_processed_test = test.iloc[:,1:int(qosElementNumber)+1].values
      #myData_processed_test[:,0] = myData_processed_test[:, 0].astype(int)
      for i in range(0,int(qosElementNumber)-1,1):         
          myData_processed_test[:,i] = myData_processed_test[:, i].astype(float) 
          
      print(myData_processed_test)



      scaler = MinMaxScaler(feature_range=(0, 1))
      myData_processed_train = scaler.fit_transform(myData_processed_train)          
      
      scaler = MinMaxScaler(feature_range=(0, 1))
      myData_processed_val = scaler.fit_transform(myData_processed_val)   

      scaler = MinMaxScaler(feature_range=(0,1))
      myData_processed_test = scaler.fit_transform(myData_processed_test)  


      # covert into input/output
      train_X, train_y = self.split_sequences(myData_processed_train, self.timesteps_n_steps_in, self.n_steps_out)
      val_X, val_y  = self.split_sequences(myData_processed_val, self.timesteps_n_steps_in, self.n_steps_out)
      test_X, test_y  = self.split_sequences(myData_processed_test, self.timesteps_n_steps_in, self.n_steps_out)
      
      
      print("SETS:")
      print(train_X.shape, train_y.shape, val_X.shape,val_y.shape,  test_X.shape, test_y.shape)    
      #vale = input("Enter your value: ")
      #print(vale)         
      return train_X, train_y, val_X, val_y,  test_X, test_y


    def print_error(self, trainY, testY, train_predict, test_predict):
        # Error of predictions
        train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
        test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
        # Print RMSE
        print('Train RMSE: %.3f RMSE' % (train_rmse))
        print('Test RMSE: %.3f RMSE' % (test_rmse))

    # Some functions to help out with
    def plot_predictions(self, test, predicted):
        plt.plot(test, color='red', label='Real QoS')
        plt.plot(predicted, color='blue', label='Predicted QoS')
        plt.title('QoS Prediction')
        plt.xlabel('Time')
        plt.ylabel('QoS')
        plt.legend()
        plt.show()

    def return_rmse(self, test, predicted):
        rmse = math.sqrt(mean_squared_error(test, predicted))
        print("The root mean squared error is {}.".format(rmse))


    def predict(self, input_dict,qosElementNumber):
        #input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
        predictions = self.model.predict(input_dict)
        prob = tf.nn.sigmoid(predictions[0])
        calc = tf.nn.softmax(predictions).numpy()
        print(predictions[0])
        print(prob[0])
        print(calc)
        maxim = 0
        pos = 0
        count = 0
        for element in calc[0]:
            if (element >= maxim):
                maxim = element
                pos = count
            count += 1

        print("The element "+str(maxim) +
              " element possition shown the calculated prediction "+str(pos))

        return pos
    
    def predictsSetupExecute(self,input,qosElementNumber):
        return self.predict([[input]],qosElementNumber)


def main():
    print("Model Setup and Test")
    csv_file = 'dataset.csv'
    data = "DNN_LSTM"
    dnn_LSTM = DNN_LSTM(data, csv_file)


if __name__ == "__main__":
    main()
