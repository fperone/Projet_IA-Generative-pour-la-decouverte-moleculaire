import pandas as pd
df = pd.read_csv("Multi-Labelled_Smiles_Odors_dataset.csv")
dicio = {}
smiles = df.iloc[:,0].tolist()
descritores = df.iloc[:,1].tolist()
if len(descritores) == len(smiles):
    print(len(descritores))
    print("True")
for nmr in range(len(smiles)):
    dicio[smiles[nmr]] = tuple(descritores[nmr].split(";"))
print(dicio)
