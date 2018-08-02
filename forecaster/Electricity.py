import csv
import time
import numpy as np

class Electricity:
    def get_data (self, path="../data/electricity.csv"):
        f = open (path, "r")
        reader = csv.reader (f)

        title_row = None
        times_arr = []
        arr = []
        for row in reader:
            if title_row == None:
                title_row = row
            else:
                times_arr.append (time.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
                arr.append ([float (i.replace (",", ".")) for i in row[1:]])
        np_arr = np.array (arr)
        transpose = np.transpose (np_arr)

        self.means = []
        self.stds = []
        for i in range (len(transpose)):
            mean = transpose[i].mean (axis=0)
            self.means.append (mean)
            transpose[i] -= mean
            std = transpose[i].std (axis=0)
            self.stds.append (std)
            transpose[i] /= std
        self.np_arr = np.transpose (transpose)
        return self.np_arr

    def encode (self, np_arr) :
        transpose = np.transpose (np_arr)
        for i in range (len (transpose)):
            transpose [i] -= self.means[i]
            transpose [i] /= self.stds [i]
        return np.transpose (transpose)

    def decode (self, np_arr):
        transpose = np.transpose (np_arr)
        for i in range (len (transpose)):
            transpose[i] *= self.stds[i]
            transpose[i] += self.means[i]
        return np.transpose (transpose)

