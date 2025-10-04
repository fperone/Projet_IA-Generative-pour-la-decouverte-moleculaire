import csv
import pandas as pd
x = input("Type the dataset filepath: ")
df = pd.read_csv("Multi-Labelled_Smiles_Odors_dataset.csv")
print(df.head())  # mostra as primeiras linhas
