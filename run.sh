#!/bin/bash


curl -o data/LD2011_2014.txt.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
unzip data/LD2011_2014.txt.zip

python3 pre-processing/ElectricityToCSV.py

python3 forecaster/QOnlineRetailRNN.py
