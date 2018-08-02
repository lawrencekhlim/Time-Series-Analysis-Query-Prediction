import csv
from models.QLearn import QLearn
from models.NaiveModel import NaiveModel
from models.EarliestModel import EarliestModel
from models.AverageModel import AverageModel
from models.RNNModel2 import RNNModel


import numpy as np
import keras.layers as L
#from keras.layers.advanced_activations import LeakyReLU
import keras.models as M
import keras.optimizers as O


DATA_DIR = "../data/timeseriesOnlineRetailCleaned2.csv"

class QOnlineRetail:
    def __init__ (self):
        self.data = []
        self.training = (0, 0.6)
        self.validation = (0.6, 0.8)
        self.testing = (0.8, 0.95)
        #self.data_size = 24
        self.data_size = 7
        
        self.num_queries = 200
        # q = 1, epochs = 100
        # q = 5, epochs = 20
        # q = 100, epochs = 10
        # q = 2000, epochs = 3
        #print (self.data)
        #self.predictor1 = QLearn(threshold=0.5, regularization=True)
        self.baseline1 = NaiveModel()
        self.baseline2 = EarliestModel()
        
        
        saved_models = [(10, "LSTM", "sgd", 0.001, 2, 32, "../results/10Qs/LSTM/2L/32HS/sgd-0.001lr/20180731-134128-683987/900epochs.h5")]
        model_num = 0
        
        
        self.num_queries = saved_models[model_num][0]
        # q = 1, epochs = 100
        # q = 5, epochs = 20
        # q = 100, epochs = 10
        # q = 2000, epochs = 3
        
        with open (DATA_DIR, 'r') as f:
            #with open ('hourlyTimeSeriesOnlineRetailCleaned.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            title_row = True
            for row in reader:
                if title_row:
                    self.products = row
                    title_row = False
                else:
                    integer_data = [int(row[i]) for i in range (self.num_queries)]
                    self.data.append (integer_data)
        
        self.predictor3 = RNNModel(saved_models[model_num][0], rnn_type=saved_models[model_num][1], optimizer_type=saved_models[model_num][2], learning_rate=saved_models[model_num][3], layers=saved_models[model_num][4], hidden_size=saved_models[model_num][5])
        self.predictor3.load_model (saved_models[model_num][6])
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
        self.predictor3.test_model (self.xval, self.yval)
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
        


        train_length = int(self.training[1]*len (self.y_data)) - int (self.training[0]*len (self.y_data))
        x_train = np.zeros ((1, train_length, self.num_queries)) # 1 example, number of time steps, number of queries
        for i in range (int(self.training[0] * len (self.x_data)), int(self.training[0] * len (self.x_data)) + train_length):
            for q in range (self.num_queries):
                x_train[0][i][q] = self.x_data[i][q] # sets the query at the time step to either 0 or 1
        
        y_train = np.zeros ((1, train_length, self.num_queries)) # 1 example, number of time steps, number of queries
        for i in range (int(self.training[0] * len (self.y_data)), int(self.training[0] * len (self.y_data)) + train_length):
            for q in range (self.num_queries):
                y_train[0][i][q] = self.y_data[i][q]

        """
            y_train = []
            for q in range (self.num_queries):
            y_train.append(np.zeros ((1, train_length, 2)))
            for i in range (int(self.training[0]*len (self.y_data)), int(self.training[0] * len (self.x_data)) + train_length):
            y_train[q][0][i][self.y_data[i][q]] = 1
        """
        
        
        #Validation
        val_length = int (self.validation[1]*len (self.x_data)) - int (self.validation[0]*len (self.x_data))
        
        x_val = np.zeros ((1, val_length, self.num_queries))
        for i in range (int(self.validation[0] * len (self.x_data)), int(self.validation[0] * len (self.x_data)) + val_length):
            for q in range (self.num_queries):
                x_val[0][i-int (self.validation[0]*len (self.x_data))][q] = self.x_data[i][q]
        
        y_val = np.zeros ((1, val_length, self.num_queries))
        for i in range (int(self.validation[0] * len (self.y_data)), int(self.validation[0] * len (self.y_data)) + val_length):
            for q in range (self.num_queries):
                y_val[0][i-int (self.validation[0]*len (self.y_data))][q] = self.y_data[i][q]
        
        self.xtrain = x_train
        self.ytrain = y_train
        self.xval = x_val
        self.yval = y_val

        self.baseline1.train(input, output)
    
        self.baseline2.train (input, output)




if __name__== "__main__":
    test = QOnlineRetail ()
    
    #test.clean_data()

    test.train_data()
    test.validate_predictor()
    #test.print_concepts()
