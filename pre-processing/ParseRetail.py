
from openpyxl import load_workbook
import csv

from datetime import timedelta
from datetime import date

table = []
reverse_table = {}
def table_index (stock_code):
    if (stock_code in reverse_table):
        return reverse_table[stock_code]
    else:
        reverse_table [stock_code] = len (table)
        table.append(stock_code)
        return reverse_table[stock_code]

def parse_date (invoice_date):
    return (invoice_date.month, invoice_date.day)
    
    '''
    date = invoice_date.split(" ")[0]
    month_day_year = date.split("/")
    return (month_day_year[0],month_day_year[1])
    '''

def return_date (invoice_date):
    return date (invoice_date.year, invoice_date.month, invoice_date.day)


print ("Loading file ...")
wb = load_workbook(filename ="OnlineRetail.xlsx")
print ("... File loaded")
sheet_ranges = wb.get_sheet_names()
first_sheet = sheet_ranges[0]
#print (sheet_ranges)
worksheet = wb.get_sheet_by_name(first_sheet)

first_day = parse_date(worksheet['E' + str (2)].value)
day_1 = return_date (worksheet['E'+str(2)].value)
print (return_date (worksheet['E'+str(2)].value))

print ("Setting up the dictionary of dates")
dict_days = {}
for i in range (0, 366):
    dict_days [day_1] = i
    day_1 = day_1 + timedelta (days=1)
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
matrix = [[0 for col in range (len (table))] for row in range (366)]
for row in range (2, 500000):
    date_of_sale = return_date (worksheet['E'+str(row)].value)
    date_index = dict_days[date_of_sale]
    stock_code_index = table_index (worksheet['B' + str(row)].value)
    matrix [date_index][stock_code_index] = 1
    if row%100 == 0:
        print ("Current row = " + str (row))
print ("Done placing into matrix")


print ("Writing into CSV")
with open ('timeseriesOnlineRetail.csv', "w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow (table)
    writer.writerows (matrix)

# 500,000


