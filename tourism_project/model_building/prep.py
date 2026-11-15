import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Dropping the 'Unnamed: 0' and 'CustomerID' columns from the dataset
columns_to_drop = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
df.drop(columns=columns_to_drop, inplace=True)

# Replacing Fe Male with Female in the Gender column
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
