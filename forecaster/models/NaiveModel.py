import numpy as np

class NaiveModel:
    def __init__ (self, threshold=0.5, regularization=False):
        self.threshold = threshold
        self.regularization = regularization
    
    def train (self, X, Y):
        Y1 = np.matrix (Y).transpose().tolist()
        y1_rows = len (Y1)
        y1_cols = len (Y1[0])
        self.output_size = y1_rows
    
    def predict (self, input):
        out = self.output_size * [0]
        for i in range (0, self.output_size):
            out[i] = input [len (input)- self.output_size+i]
        return out


    def test_model (self, input, output, verbose=True):
        total_l2_error = 0
        total_l1_error = 0
        predictions = []
        
        for i in range (len (output)):
            prediction = self.predict (input[i])
            predictions.append (prediction)
            err = self.l2_error (output[i], prediction)
            total_l2_error += err
            err = self.l1_error (output[i], prediction)
            total_l1_error += err
            if verbose:
                print (str (i+1)+ ") ")
                print ("\tActual\t\tPredicted")
                for prod in range (0, len (output[i])):
                    print ("\t"+str(output [i][prod])+"\t\t"+str(prediction[prod]))
                #print ("Actual:    " + str(output[i]))
                #print ("Predicted: "+ str(prediction))
                print ("L2 Error:  " + str (err))
                print ("Std Dev:   " + str ((err/len (output[i])) ** (0.5)))
                print ("")
        total_l2_error = total_l2_error / len (output)
        print ("Average Error L2: " + str (total_l2_error))
        
        rmsd = (total_l2_error / len (output[0]))**0.5
        print ("RMSD: " + str(rmsd))
        
        total_l1_error = total_l1_error / len (output)
        print ("Average Error L1: " + str (total_l2_error))
        
        average_deviation = (total_l1_error / len (output[0]))
        print ("Average Deviation per Query: " + str(average_deviation))
        
        #coeffs = self.coeff_of_determination (output, predictions)
        #print ("Coefficients of determination: " + str (coeffs))
        
        cntng_table = self.contingency_table (output, predictions)
        print (cntng_table[0])
        
    def l1_error (self, real, prediction):
        error = 0
        for i in range (len (real)):
            error += abs(real[i] - prediction[i])
        return error

    def l2_error (self, real, prediction):
        error = 0
        for i in range (len (real)):
            error += (real[i] - prediction[i]) ** 2
        return error

    def mean (self, list):
        return sum (list) / len (list)
    
    def variance (self, list):
        mean = self.mean (list)
        variance = 0
        for value in list:
            variance += (value - mean) ** 2
        return variance / len(list)

    """
    Returns a contingency table
    """
    def contingency_table (self, real, prediction):
        table = []
        for i in range (0, len (real[0])):
            table.append ([0, 0, 0, 0])
            for b in range (0, len (real)):
                if (real[b][i] == 1):
                    if (prediction [b][i] >= self.threshold):
                        table[i][0] += 1 # True positive
                    else:
                        table[i][2] += 1 # False negative
                else:
                    if (prediction[b][i] >= self.threshold):
                        table[i][1] += 1 # False positive
                    else:
                        table[i][3] += 1 # True negative
        return table


