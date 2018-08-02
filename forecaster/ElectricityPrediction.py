import csv
from models.QLearn import QLearn
from models.NaiveModel import NaiveModel
from models.EarliestModel import EarliestModel
from models.AverageModel import AverageModel
from models.RNNModel3 import RNNModel

from Electricity import Electricity
import numpy as np
import keras.layers as L
#from keras.layers.advanced_activations import LeakyReLU
import keras.models as M
import keras.optimizers as O


DATA_DIR = "../data/timeseriesOnlineRetailCleaned2.csv"

class ElectricityPrediction:
    def __init__ (self):
        self.data = []
        self.training = (0, 0.5)
        self.validation = (0.5, 0.8)
        self.testing = (0.8, 0.95)
        #self.data_size = 24
        self.data_size = 96
        
        
        self.num_queries = 10
        # q = 1, epochs = 100
        # q = 5, epochs = 20
        # q = 100, epochs = 10
        # q = 2000, epochs = 3
        
        self.ds = Electricity ()
        data = self.ds.get_data()
        transpose = np.transpose (data)
        transpose = transpose [:self.num_queries]
        data = np.transpose (transpose)
        self.training_data = data[int (self.training[0]* len (data)): int(self.training[1]* len (data))]
        self.validation_data = data[int (self.validation[0]* len (data)): int(self.validation[1]* len (data))]
        self.testing_data = data[int (self.training[0] * len (data)): int(self.training[1] * len (data))]
        #for i in range (int(len (transpose) / self.num_queries)):
        
        
        
        #print (self.data)
        self.predictor1 = QLearn(threshold=0.5, regularization=False)
        self.baseline1 = NaiveModel()
        self.baseline2 = EarliestModel()
        self.predictor3 = RNNModel(self.num_queries, rnn_type="LSTM", optimizer_type="adam", learning_rate=0.001, layers=2, hidden_size=64, recurrent_dropout=0.2)
        #self.baseline3 = AverageModel(threshold=0.75, regularization=True)
    
   
        
    
    
    def validate_predictor (self):
        input = []
        output = []
        
        validation_data = self.validation_data.tolist()
        week_data = []
        for days in range (0, self.data_size):
            today = validation_data[days]
            week_data = week_data+ today
    
    
        for days in range (self.data_size, len (self.validation_data)):
            today = validation_data[days]
            input.append (week_data)
            output.append (today)
            week_data = week_data+ today
            week_data = week_data[len (self.validation_data[0]):]

        
        print ()
        print ("Linear Algebra Model")
        #self.predictor1.try_ktruncations (input,output)
        self.predictor1.test_model (input, output, verbose=False)
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
        self.predictor3.test_model (self.xval, self.yval)
        #self.predictor2.test_model_keras (input, output)
        #self.predictor2.test_model(input, output, verbose=False)
                

    
    def train_data (self):
        input = []
        output = []
        
        training_data = self.training_data.tolist()
        week_data = []
        for days in range (0, self.data_size):
            today = training_data[days]
            week_data = week_data+ today
        
        
        for days in range (self.data_size, len (self.training_data)):
            today = training_data[days]
            input.append (week_data)
            output.append (today)
            week_data = week_data+ today
            week_data = week_data [len (self.training_data[0]):]

        self.predictor1.set_training_data (input, output)
        
        
        self.x_data = self.data[:-1]
        self.y_data = self.data[1:]
        
        x_train = np.array ([self.training_data[:-1]])
        y_train = np.array ([self.training_data[1:]])
        
        x_val = np.array ([self.validation_data[:-1]])
        y_val = np.array ([self.validation_data[1:]])

        self.xtrain = x_train
        self.ytrain = y_train
        self.xval = x_val
        self.yval = y_val
        
        print ("")
        print ("Window size: " + str (self.data_size))
        print ("")
        print ("Training Model...")
        self.predictor1.train()
        
        self.baseline1.train(input, output)
        
        self.baseline2.train (input, output)
        
        self.predictor3.train (x_train, y_train, x_val, y_val)

        print ("... Done Training")





if __name__== "__main__":
    test = ElectricityPrediction ()
    
    #test.clean_data()

    test.train_data()
    test.validate_predictor()
    #test.print_concepts()
