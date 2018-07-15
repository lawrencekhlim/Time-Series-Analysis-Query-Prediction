import csv
from QLearn import QLearn
from Model import NaiveModel, EarliestModel
from AverageModel import AverageModel
#from RNNModel4 import RNNModel
#from GRUModel import GRUModel


import numpy as np
import keras.layers as L
#from keras.layers.advanced_activations import LeakyReLU
import keras.models as M
import keras.optimizers as O

import numpy


class QOnlineRetail:
    def __init__ (self):
        self.data = []
        self.training = (0, 0.6)
        self.validation = (0.6, 0.8)
        self.testing = (0.8, 1)
        #self.data_size = 24
        self.data_size = 7
        
        with open ('timeseriesOnlineRetailCleaned2.csv', 'r') as f:
            #with open ('hourlyTimeSeriesOnlineRetailCleaned.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            title_row = True
            for row in reader:
                if title_row:
                    self.products = row
                    title_row = False
                else:
                    integer_data = [int(row[0])]
                    self.data.append (integer_data)
        #print (self.data)
        #self.predictor1 = QLearn(threshold=0.5, regularization=True)
        #self.predictor2 = RNNModel ()
        self.baseline1 = NaiveModel()
        self.baseline2 = EarliestModel()
        self.predictor3 = simple_model((None, len(self.data[0])), 2)
        #self.baseline3 = AverageModel(threshold=0.75, regularization=True)
    
   
        
    
    
    def validate_predictor (self):
        input = []
        output = []
        
        week_data = []
        for days in range (int (360*self.validation[0]), int (360*self.validation[0])+self.data_size):
            today = self.data[days]
            week_data = week_data+ today
    
    
        for days in range (int (360*self.validation[0])+self.data_size, int(360*self.validation[1])):
            today = self.data[days]
            input.append (week_data)
            output.append (today)
            week_data = week_data+ today
            for i in range (len (self.data[0])):
                week_data.pop (0)
        
        #print ()
        #print ("Linear Algebra Model")
        #self.predictor1.try_ktruncations (input,output)
        #self.predictor1.test_model (input, output, verbose=False)
        #self.predictor1.print_concepts()
        
        print ()
        print ("Previous Day Naive Model")
        self.baseline1.test_model (input, output, verbose=False)
        
        print ()
        print ("Earliest Day Naive Model")
        self.baseline2.test_model (input, output, verbose=False)
        
        #print ()
        #print ("Average of Past Days Model")
        #self.baseline3.test_model (input, output, verbose=False)
        
    
    
        print ()
        print ("RNN Model")
        test_model (self.predictor3, self.xval, self.yval)
        #self.predictor2.test_model_keras (input, output)
        #self.predictor2.test_model(input, output, verbose=False)
                

    
    def train_data (self):
        
        
        
        input = []
        output = []
        
        week_data = []
        for days in range (int (360*self.training[0]), int (360*self.training[0])+self.data_size):
            today = self.data[days]
            week_data = week_data+ today
        
        
        for days in range (int (360*self.training[0])+self.data_size, int(360*self.training[1])):
            today = self.data[days]
            input.append (week_data)
            output.append (today)
            week_data = week_data+ today
            for i in range (len (self.data[0])):
                week_data.pop (0)
        #self.predictor1.set_training_data (input, output)
        
        
        self.x_data = self.data[:-1]
        self.y_data = self.data[1:]
        
        
        x_train = []
        x_train.append(self.x_data [int(self.training[0] * len (self.x_data)): int(self.training[1] * len (self.x_data)) ])
        x_train = np.array (x_train)
        
        train_length = int(self.training[1]*len (self.y_data)) - int (self.training[0]*len (self.y_data))
        y_train = np.zeros ((1, train_length, 1*2))
        for i in range (int(self.training[0]*len (self.y_data)), int (self.training[1]*len (self.y_data))):
            y_train [0][i-int (self.training[0]*len (self.y_data))][self.y_data[i]] = 1
        #y_train = []
        #y_train.append(self.y_data [int(self.training[0] * len (self.y_data)): int(self.training[1] * len (self.y_data) )])
        #y_train = np.array (y_train)
        

        x_val = []
        x_val.append(self.x_data [int(self.validation[0] * len (self.x_data)): int(self.validation[1] * len (self.x_data) )])
        x_val = np.array (x_val)
        
        y_val = np.zeros ((1, int(self.validation[1]*len (self.y_data)) - int (self.validation[0]*len (self.y_data)), 1*2))
        for i in range (int(self.validation[0]*len (self.y_data)), int (self.validation[1]*len (self.y_data))):
            y_val [0][i-int(self.validation[0] * len (self.x_data))][self.y_data[i]] = 1
        #y_val = []
        #y_val.append(self.y_data [int(self.validation[0] * len (self.y_data)): int(self.validation[1] * len (self.y_data) )])
        #y_val = np.array (y_val)
        
        
        print ("")
        print ("Window size " + str (self.data_size) + " days")
        print ("")
        print ("Training Model...")
        #self.predictor1.train()
        
        self.baseline1.train(input, output)
        
        self.baseline2.train (input, output)
        
        train (self.predictor3, x_train, y_train, x_val, y_val)

        self.xtrain = x_train
        self.ytrain = y_train
        self.xval = x_val
        self.yval = y_val

        
        #self.baseline3.train (input, output)
        
        #self.predictor2.train (input, output)
        print ("... Done Training")


    #def print_concepts (self):
    #self.predictor1.print_concepts()


def train (model, x_train, y_train, x_val, y_val):
    BATCH_SIZE=1
    # Train the model each generation and show predictions against the validation
    # dataset.
    
    print (y_val)
    print (len (y_val[0]))
    
    
    for iteration in range(1, 250):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=1,
                  validation_data=(x_val, y_val))
        # Select 10 samples from the validation set at random so we can visualize
        # errors.
        #ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([0])], y_val[np.array([0])]
        preds = model.predict(rowx, verbose=0)
        print (preds)
        print (len (preds[0]))
        y_classes = preds.argmax(axis=-1)
        print (y_classes)
        #q = ctable.decode(rowx[0])
        #correct = ctable.decode(rowy[0])
        #guess = ctable.decode(preds[0], calc_argmax=False)
        #print('Q', q[::-1] if REVERSE else q, end=' ')
        #print('T', correct, end=' ')
        #if correct == guess:
        #    print(colors.ok + '☑' + colors.close, end=' ')
        #else:
        #    print(colors.fail + '☒' + colors.close, end=' ')
        #    print(guess)

def simple_model (input_shape, num_output_categories):
    RNN = L.SimpleRNN
    HIDDEN_SIZE = 7
    LAYERS = 2

    print('Build model...')
    print (input_shape)
    model = M.Sequential()

    model.add(RNN(HIDDEN_SIZE, input_shape=(None, 1),return_sequences=True))

    for _ in range(LAYERS-1):
        model.add(RNN(HIDDEN_SIZE, input_shape=(None, 7), return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    model.add(L.TimeDistributed(L.Dense(num_output_categories)))
    model.add(L.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model


def test_model (model, input, output, verbose=False):
    total_l2_error = 0
    total_l1_error = 0
    #predictions = []
    
    
    data_x = []
    data_point_x = []
    preds = model.predict (input)
    prediction = preds.argmax(axis=-1)
    #predictions.append (prediction)
    real = output.argmax (axis=-1)

    err = l2_error (real[0], prediction[0])
    total_l2_error += err
    err = l1_error (real[0], prediction[0])
    total_l1_error += err
    if verbose:
        print ("\tActual\t\tPredicted")
        for prod in range (0, len (output[i])):
            print ("\t"+str(output [i][prod])+"\t\t"+str(prediction[prod]))
        #print ("Actual:    " + str(output[i]))
        #print ("Predicted: "+ str(prediction))
        print ("L2 Error:  " + str (err))
        print ("Std Dev:   " + str ((err/len (real)) ** (0.5)))
        print ("")
    total_l2_error = total_l2_error / len (real[0])
    print ("Average Error L2: " + str (total_l2_error))
    
    rmsd = (total_l2_error / len (real))**0.5
    print ("RMSD: " + str(rmsd))
    
    total_l1_error = total_l1_error / len (real[0])
    print ("Average Error L1: " + str (total_l2_error))
    
    average_deviation = (total_l1_error / len (real))
    print ("Average Deviation per Query: " + str(average_deviation))
    
    #coeffs = self.coeff_of_determination (output, predictions)
    #print ("Coefficients of determination: " + str (coeffs))
    
    cntng_table = contingency_table (real, prediction)
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
    for i in range (0, len (real)):
        table.append ([0, 0, 0, 0])
        for b in range (0, len (real[0])):
            if (real[i][b] == 1):
                if (prediction [i][b] >= 0.5):
                    table[i][0] += 1 # True positive
                else:
                    table[i][2] += 1 # False negative
            else:
                if (prediction[i][b] >= 0.5):
                    table[i][1] += 1 # False positive
                else:
                    table[i][3] += 1 # True negative
    return table


if __name__== "__main__":
    test = QOnlineRetail ()
    
    #test.clean_data()

    test.train_data()
    test.validate_predictor()
    #test.print_concepts()
