
import csv

f1 = open ("../data/LD2011_2014.txt", "r")
semicolonin = csv.reader(f1, delimiter=';')

f2 = open ("../data/electricity.csv", "w")
commaout = csv.writer(f2, delimiter=',')
for row in semicolonin:
    commaout.writerow(row)
f1.close()
f2.close()
