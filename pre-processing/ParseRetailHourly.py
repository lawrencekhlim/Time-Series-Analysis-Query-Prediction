
from openpyxl import load_workbook
import csv

from datetime import timedelta
from datetime import datetime

table = []
reverse_table = {}
def table_index (stock_code):
    if (stock_code in reverse_table):
        return reverse_table[stock_code]
    else:
        reverse_table [stock_code] = len (table)
        table.append(stock_code)
        return reverse_table[stock_code]


def return_datetime (invoice_date):
    return datetime (invoice_date.year, invoice_date.month, invoice_date.day, invoice_date.hour)


print ("Loading file ...")
wb = load_workbook(filename ="OnlineRetail.xlsx")
print ("... File loaded")
sheet_ranges = wb.get_sheet_names()
first_sheet = sheet_ranges[0]
#print (sheet_ranges)
worksheet = wb.get_sheet_by_name(first_sheet)


day_1 = return_datetime (worksheet['E'+str(2)].value)
print (day_1)

print ("Setting up the dictionary of dates and hours")
dict_days = {}
for i in range (0, 366*24):
    dict_days [day_1] = i
    day_1 = day_1 + timedelta (hours=1)
print dict_days


print ("Setting up the dictionary of stockcodes")
for row in range (2, 500000):
    table_index (worksheet['B' + str(row)].value)
    if row%100 == 0:
        print ("Current row = " + str (row))
print ("Number of items: " + str (len (table)))


print ("Writing into CSV")
with open ('listOnlineRetail.csv', "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow (table)



print ("Placing into matrix")
matrix = [[0 for col in range (len (table))] for row in range (366*24)]
for row in range (2, 500000):
    time_of_sale = return_datetime (worksheet['E'+str(row)].value)
    time_index = dict_days[time_of_sale]
    stock_code_index = table_index (worksheet['B'+str(row)].value)
    matrix [time_index][stock_code_index] = 1
    if row%100 == 0:
        print ("Current row = " + str (row))
print ("Done placing into matrix")


print ("Writing into CSV")
with open ('hourlyTimeSeriesOnlineRetail.csv', "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow (table)
    writer.writerows (matrix)

# 500,000


