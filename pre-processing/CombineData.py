
import csv
import os

DATA_FOLDER_PATH = "../data/"
WORDS_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "words/")
NEW_WORDS_PATH = os.path.join (DATA_FOLDER_PATH, "words.csv")

file_name_0 = os.path.join (WORDS_FOLDER_PATH, "0.csv")


with open (file_name_0, 'r') as f:
    reader = csv.reader (f)
    large_list = list (reader)


for i in range (1, 100):
    file_name = os.path.join (WORDS_FOLDER_PATH, str(i)+".csv")
    with open (file_name, 'r') as f:
        reader = csv.reader (f)
        tmp_list = list(reader)

    large_list = [large_list[i] + [tmp_list[i][1]] for i in range (len(large_list))]

with open (NEW_WORDS_PATH, 'w') as f:
    writer = csv.writer (f)
    writer.writerows (large_list)
