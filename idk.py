import csv
import pandas as pd
x = input("Type the dataset filepath: ")
df = pd.read_csv("x")
print(df.head())  # mostra as primeiras linhas
