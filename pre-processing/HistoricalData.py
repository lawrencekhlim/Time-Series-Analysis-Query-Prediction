
from pytrends.request import TrendReq
#import pandas
import time
import os

DATA_FOLDER_PATH = "../data/words/"
paths = ["words.txt", "census-dist-female-first.txt"]

words_file = open (paths[0], "r")

pytrends = TrendReq (hl='en-US')


for i in range (100):
    #word = words_file.readline ().split()[0].strip()
    #word = word[0] + word[1:].lower()
    word = words_file.readline().strip()
    print ("Obtaining historical data for \"" +word+ "\"")
    pandas_obj = pytrends.get_historical_interest ([ word], year_start=2018, month_start=4, day_start=1, hour_start=0, year_end=2018, month_end=6, day_end=10, hour_end=0, geo='US')
    pandas_obj = pandas_obj.drop (columns=["isPartial"])
    csv_path = os.path.join (DATA_FOLDER_PATH, str(i) +'.csv')
    with open (csv_path, "w") as f:
        pandas_obj.to_csv (f)
    time.sleep (3)

words_file.close()
