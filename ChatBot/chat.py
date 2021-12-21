import sys
import json
import torch
from model import NeuralNet
from tokenizer import bagOfWords, tokenize
import random


try:
    fileName = sys.argv[-1]
    data = torch.load(fileName)
    print(f"data is: {sys.argv[1]}")
except:
    print("need an parameter ojbect... type .pth file too!    it must be named as data.pth")
    exit()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('textbook.json', 'r') as json_data:
    intents = json.load(json_data)

inptSize = data["inptSize"]
hidnSize = data["hidnSize"]
ouptSize = data["ouptSize"]
allWords = data['allWords']
tags = data['tags']
modelState = data["modelState"]

model = NeuralNet(inptSize, hidnSize, ouptSize).to(device)
model.load_state_dict(modelState)
model.eval()

botName = input("My name is?: ") + ": "
print(f"{botName}nice to meet you!")
while 1:
    sentence = input("sent: ")
    if sentence == "exit":
        exit()

    sentence = tokenize(sentence)
    case = bagOfWords(sentence, allWords)
    case = case.reshape(1, case.shape[0])
    case = torch.from_numpy(case)

    output = model(case)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f"{botName}: I don't know what to say...")
