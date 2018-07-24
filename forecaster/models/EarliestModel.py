from .NaiveModel import NaiveModel

class EarliestModel (NaiveModel):
    
    def predict (self, input):
        out = self.output_size * [0]
        for i in range (0, self.output_size):
            out[i] = input [i]
        return out
