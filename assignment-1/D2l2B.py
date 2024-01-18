# D2l2B.py D2l 2.2 2023 cheng CS5173/6073
# Usage: python D2l2B.py

# D2l 2.2 Data Preprocessing
# D2l 2.2.1 Reading the Dataset
import os
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
import pandas as pd
data = pd.read_csv(data_file)
print(data)

# D2l 2.2.2 Data Preparation
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
inputs = inputs.fillna(inputs.mean())
print(inputs)

# D2l 2.2.3 Conversion to the Tensor Format
import torch
X, y = torch.tensor(inputs.values), torch.tensor(targets.values)
print(X, y)
