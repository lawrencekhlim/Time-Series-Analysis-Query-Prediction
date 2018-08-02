import random
import csv

DATA_DIR = "../data/timeseriesOnlineRetailCleaned2.csv"



title_row = True
all_data = []

int_data = []
with open ('../data/timeseriesOnlineRetailCleaned2.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        all_data.append ([i for i in row])
        if not title_row:
            int_data.append ([int(i) for i in row])
        title_row = False

#print (all_data)
sums_of_columns = [ sum(x) for x in zip(*int_data) ]
#print (sums_of_columns)
all_data_transpose = list(map(list,zip(*all_data)))
#print (all_data_transpose)
cleaned_data_transpose = []

for i in range (1, len (sums_of_columns)):
    if sums_of_columns[i] >= 20:
        cleaned_data_transpose.append (all_data_transpose[i])
#print (cleaned_data_transpose)
random.shuffle (cleaned_data_transpose)

cleaned_data = list(map( list, zip (*([all_data_transpose[0]] + cleaned_data_transpose)) ))

with open ('../data/randomizedOnlineRetail.csv', "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows (cleaned_data)
