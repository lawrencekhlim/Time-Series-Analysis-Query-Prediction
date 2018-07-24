import csv
from models import QLearn
from models import NaiveModel, EarliestModel
from models import AverageModel
#from RNNModel import RNNModel
#from GRUModel import GRUModel

class QOnlineRetail:
    def __init__ (self):
        self.data = []
        self.training = (0, 0.6)
        self.validation = (0.6, 0.8)
        self.testing = (0.8, 1)
        self.data_size = 7
        
        self.num_queries = [10, 50, 75, 100, 200, 300, 500, 750, 1000, 1500, 2000]
        
        with open ('timeseriesOnlineRetailCleaned2.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            title_row = True
            for row in reader:
                if title_row:
                    self.products = row
                    title_row = False
                else:
                    integer_data = [int(i) for i in row]
                    self.data.append (integer_data)
        #print (self.data)
        self.predictor1 = QLearn(threshold=0.5, regularization=True)
        #self.predictor2 = GRUModel ()
        self.baseline1 = NaiveModel()
        self.baseline2 = EarliestModel()
        self.baseline3 = AverageModel(threshold=0.75, regularization=True)

        
    
    
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
        
        print ()
        print ("Linear Algebra Model")
        self.predictor1.try_ktruncations (input,output)
        self.predictor1.test_model (input, output, verbose=False)
        #self.predictor1.print_concepts()
        
        print ()
        print ("Previous Day Naive Model")
        self.baseline1.test_model (input, output, verbose=False)
        
        print ()
        print ("Earliest Day Naive Model")
        self.baseline2.test_model (input, output, verbose=False)
        
        print ()
        print ("Average of Past Days Model")
        self.baseline3.test_model (input, output, verbose=False)
        

    
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
        self.predictor1.set_training_data (input, output)
        
        print ("")
        print ("Window size " + str (self.data_size) + " days")
        print ("")
        print ("Training Model...")
        self.predictor1.train()
        
        self.baseline1.train(input, output)
        
        self.baseline2.train (input, output)
        
        self.baseline3.train (input, output)
        
        #self.predictor2.train (input, output)
        print ("... Done Training")


    def print_concepts (self):
        self.predictor1.print_concepts()


if __name__== "__main__":
    test = QOnlineRetail ()
    test.train_data()
    test.validate_predictor()
