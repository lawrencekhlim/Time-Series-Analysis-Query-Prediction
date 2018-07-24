
import numpy as np
import keras.layers as L
import keras.models as M
import keras.optimizers as O
from keras import Input


import pickle
import matplotlib.pyplot as plt

RESULTS_FOLDER = "../results/"

class RNNModel:
    def __init__(self,  num_queries, rnn_type="LSTM", optimizer_type="adam", learning_rate=0.001):
        
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

        
        self.HIDDEN_SIZE = 1024 # 4, 16, 32
        self.LAYERS = 1 # 1, 2, 4, layers
        self.NUM_QUERIES = num_queries

        #lr = 0.001 or 0.01
        print('Build model...')
        #print (input_shape)
        
        input = Input(shape=(None, num_queries))
        x = RNN (self.HIDDEN_SIZE, return_sequences=True) (input)
        for _ in range (self.LAYERS-1):
            x = RNN (self.HIDDEN_SIZE, return_sequences=True) (x)
        query_prediction = []
        for i in range (0, num_queries):
            query_prediction.append (L.Dense(2,  activation='softmax', name=str(i))(x))
        self.model = M.Model (input, query_prediction)
        
        self.model.compile (optimizer='sgd', loss=['categorical_crossentropy' for i in range (num_queries)])
        
        self.model.summary ()
        


    def train (self, x_train, y_train, x_val, y_val, save=True):
        BATCH_SIZE=1
        
        
        
        history = self.model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=100,
                    validation_data=(x_val, y_val))

        if save:
            history_dict = history.history
        
            file_name = RESULTS_FOLDER + str(self.NUM_QUERIES) +"Qs-"+ self.RNN_TYPE + "-" + str (self.LAYERS)+"L-" + str (self.HIDDEN_SIZE)+"HS-"+self.OPTIMIZER_TYPE +"-"+str(self.LEARNING_RATE)+"lr"  +".pkl"
            f = open (file_name, "wb")
            pickle.dump (history_dict, f)
            f.close()
    
        #self.predict (x_val)
    
    def predict (self, input):
        out = self.model.predict (input)
        #print (out)
        return self.decode (out)
                        
    def decode (self, encoded):
        #print (self.NUM_QUERIES)
        #print (len (encoded))
        #print (len (encoded[0]))
        #print (len (encoded[0][0]))
        #print (len (encoded[0][0][0]))
        decode = [[[0 for q in range (len (encoded))] for i in range (len (encoded[0][0]))] for j in range (len (encoded[0]))]
        for j in range (len (encoded[0])):
            for i in range (len (encoded[0][0])):
                for q in range (len (encoded)):
                    if encoded[q][j][i][1] >= 0.5:
                        decode[j][i][q] = 1
        return decode

    def test_model (self, input, output):
        total_l2_error = 0
        total_l1_error = 0
        #predictions = []
        
        
        data_x = []
        data_point_x = []

        prediction = self.predict (input)
        prediction = np.array(prediction) [:, 8:]
        #print (output)
        real = self.decode (output)
        real = np.array(real)[:, 8:]
        #print (prediction)
        #print (real)
        for j in range (len (real)):
            for i in range (len (real[j])):
                err = l2_error (real[j][i], prediction[j][i])
                total_l2_error += err
                err = l1_error (real[j][i], prediction[j][i])
                total_l1_error += err
        #print (len(real))
        #print (len(real[0]))
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
        print (cntng_table[0])


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


