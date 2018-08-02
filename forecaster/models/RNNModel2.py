
import numpy as np
import keras.layers as L
import keras.models as M
from keras.models import load_model
import keras.optimizers as O
from keras import Input

import gc
import os
import datetime
import pickle
import matplotlib.pyplot as plt

RESULTS_FOLDER = "/Users/LawrenceLim/Documents/Programming/CleanedResearch/results/"

class RNNModel:
    def __init__(self,  num_queries, rnn_type="LSTM", optimizer_type="adam", learning_rate=0.001, layers=2, hidden_size=32, recurrent_dropout=0.2):
        
        self.RNN_TYPE = rnn_type
        if self.RNN_TYPE == "LSTM":
            RNN = L.LSTM
        elif self.RNN_TYPE == "GRU":
            RNN = L.GRU
        else:
            RNN = L.SimpleRNN
        
        self.LEARNING_RATE = learning_rate
        self.OPTIMIZER_TYPE = optimizer_type
        if self.OPTIMIZER_TYPE == "sgd":
            opt = O.SGD (lr=self.LEARNING_RATE)
        elif self.OPTIMIZER_TYPE == "adam":
            opt = O.Adam (lr=self.LEARNING_RATE)
        elif self.OPTIMIZER_TYPE == "rmsprop":
            opt = O.RMSprop (lr=self.LEARNING_RATE)

        
        self.HIDDEN_SIZE = hidden_size # 4, 16, 32
        self.LAYERS = layers # 1, 2, 4, layers
        self.NUM_QUERIES = num_queries

        #lr = 0.001 or 0.01
        print('Build model...')
        #print (input_shape)
        
        input = Input(shape=(None, num_queries))
        x = RNN (self.HIDDEN_SIZE, return_sequences=True, recurrent_dropout=recurrent_dropout) (input)
        for _ in range (self.LAYERS-1):
            x = RNN (self.HIDDEN_SIZE, return_sequences=True, recurrent_dropout=recurrent_dropout) (x)
        """
        query_prediction = []
        for i in range (0, num_queries):
            query_prediction.append (L.Dense(2,  activation='softmax', name=str(i))(x))
        self.model = M.Model (input, query_prediction)
        self.model.compile (optimizer=opt, loss=['categorical_crossentropy' for i in range (num_queries)])
        """
        x = L.Dense(self.NUM_QUERIES, activation='sigmoid') (x)
        self.model = M.Model (input, x)
        self.model.compile (optimizer=opt, loss='binary_crossentropy')
        self.model.summary ()
        


    def train (self, x_train, y_train, x_val, y_val, save=True):
        BATCH_SIZE=1
        total_epochs = 1000
        epochs_per_round = 20
        rounds = int (total_epochs/epochs_per_round)
        
        timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        dir_arr = [str(self.NUM_QUERIES) +"Qs", self.RNN_TYPE,str (self.LAYERS)+"L", str (self.HIDDEN_SIZE)+"HS",self.OPTIMIZER_TYPE +"-"+str(self.LEARNING_RATE)+"lr", timestr]
        
        dir_path = RESULTS_FOLDER
        for path in dir_arr:
            dir_path = os.path.join (dir_path, path)
            if not os.path.exists (dir_path):
                os.mkdir (dir_path)

        history_dict = {}
        for i in range (rounds):
            print ("Epochs: " + str ((i)*epochs_per_round))
            history = self.model.fit (x_train, y_train, batch_size=BATCH_SIZE, epochs=epochs_per_round, validation_data=(x_val, y_val))
            
            
            
            self.model.save(os.path.join(dir_path, str((i+1)*epochs_per_round)+"epochs.h5" ))
                
                
            for key in history.history.keys():
                if not key in history_dict:
                    history_dict[key] = history.history[key]
                else:
                    history_dict[key] += history.history[key]
            
            if ((i+1)%10 == 0):
                file_name = os.path.join(dir_path, str((i+1)*epochs_per_round) +"history.pkl")
                f = open (file_name, "wb")
                pickle.dump (history_dict, f)
                f.close()
            
        """
        history = self.model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=150,
                    validation_data=(x_val, y_val))
        history_dict = history.history
        """
                                      
        if save:
        
            file_name = os.path.join(dir_path, "history.pkl")
            f = open (file_name, "wb")
            pickle.dump (history_dict, f)
            f.close()
        
        del history
        #self.predict (x_val)
    
    
    def predict (self, input):
        out = self.model.predict (input)
        return out
        #return self.decode (out)
    
    """"
    def decode (self, encoded):
        decode = [[[0 for q in range (len (encoded))] for i in range (len (encoded[0][0]))] for j in range (len (encoded[0]))]
        for j in range (len (encoded[0])):
            for i in range (len (encoded[0][0])):
                for q in range (len (encoded)):
                    if encoded[q][j][i][1] >= 0.5:
                        decode[j][i][q] = 1
        return decode
    """
    
    def delete_model (self):
        del self.model
        gc.collect()
    
    def load_model (self, path):
        self.model = load_model (path)

    def test_model (self, input, output):
        total_l2_error = 0
        total_l1_error = 0
        #predictions = []
        
        
        data_x = []
        data_point_x = []

        prediction = self.predict (input)
        prediction = np.array(prediction) [:, 8:]
        #print (output)
        #real = self.decode (output)
        real = np.array(output)[:, 8:]
        print (prediction)
        print (real)
        #print (prediction)
        #print (real)
        for j in range (len (real)):
            for i in range (len (real[j])):
                err = l2_error (real[j][i], prediction[j][i])
                total_l2_error += err
                err = l1_error (real[j][i], prediction[j][i])
                total_l1_error += err
        print (len(real))
        print (len(real[0]))
        print (len(real[0][0]))
        
        total_l2_error = total_l2_error / (len (real[0])) / len (real)
        print ("Average Error L2: " + str (total_l2_error))

        rmsd = (total_l2_error / len (real[0][0]))**0.5
        print ("RMSD: " + str(rmsd))

        total_l1_error = total_l1_error / (len (real[0])) / len (real)
        print ("Average Error L1: " + str (total_l2_error))
    
        average_deviation = (total_l1_error / len (real[0][0]))
        print ("Average Deviation per Query: " + str(average_deviation))
    
    
        cntng_table = contingency_table (real[0], prediction[0])
        print ("[TP, FP, FN, TN]")
        print (cntng_table)


def l1_error (real, prediction):
    error = 0
    for i in range (len (real)):
        error += abs(real[i] - prediction[i])
    return error

def l2_error (real, prediction):
    error = 0
    for i in range (len (real)):
        error += (real[i] - prediction[i]) ** 2
    return error

def mean (list):
    return sum (list) / len (list)

def variance (list):
    mean = self.mean (list)
    variance = 0
    for value in list:
        variance += (value - mean) ** 2
    return variance / len(list)

"""
    Returns a contingency table
    """
def contingency_table (real, prediction):
    table = []
    for i in range (0, len (real[0])):
        table.append ([0, 0, 0, 0])
        for b in range (0, len (real)):
            if (real[b][i] == 1):
                if (prediction [b][i] >= 0.5):
                    table[i][0] += 1 # True positive
                else:
                    table[i][2] += 1 # False negative
            else:
                if (prediction[b][i] >= 0.5):
                    table[i][1] += 1 # False positive
                else:
                    table[i][3] += 1 # True negative
    return table


