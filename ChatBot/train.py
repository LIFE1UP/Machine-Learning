import json
from tokenizer import tokenize, bagOfWords
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import NeuralNet


# load our train data file
with open('textbook.json', 'r', encoding='utf-8') as f:
    trainData = json.load(f)

allWords = []
tags = []
xy = []

for intent in trainData['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        tk_sentence = tokenize(pattern)
        allWords.extend(tk_sentence)
        xy.append((tk_sentence, intent['tag']))

allWords = sorted(allWords)
tags = sorted(tags)

x = []
y = []

for (pattern_sentence, tag) in xy:
    bag = bagOfWords(pattern_sentence, allWords)
    x.append(bag)
    y.append(tags.index(tag))  # CrossEntropyLoss

x = np.array(x)
y = np.array(y)
y = torch.from_numpy(y)
y = y.type(torch.long)

class myTrainSet(Dataset):
    def __init__(self):
        self.n_samples = len(x)
        self.x_data = x
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

mySet = myTrainSet()
trainData = DataLoader(dataset=mySet, batch_size=8, shuffle=True, num_workers=0)

inptSize = x.shape[1]
ouptSize = len(tags)
hidnSize = 10

model = NeuralNet(inptSize, hidnSize, ouptSize)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device=device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), 0.001)

print("Progress Bar: ", end="")
iterations = 1000
for epoch in range(iterations):
    for (words, labels) in trainData:
        predictY = model.forward(words)
        loss = criterion(predictY, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

    if epoch % (iterations / 100) == 0:
        print("#", end="")
print(f"\ntrain is over, loss: {loss.item():.4f}")

data = {
"modelState": model.state_dict(),
"inptSize": inptSize,
"hidnSize": hidnSize,
"ouptSize": ouptSize,
"allWords": allWords,
"tags": tags
}

fileName = "data.pth"
torch.save(data, fileName)

print(f"everything completed, file saved as {fileName}")
