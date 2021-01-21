---
title: Otto Dataset Analysis
description: yay
publishDate: 2020-04-21
images:
  - /200430_otto/logo.png
tags: ["python", "data analysis", "neural network", "linear model", "kaggle"]
draft: false

---
As a first project, I decided to pick a random dataset from Kaggle to analyse. In this project, I analysed project distributions from the _Otto Group_ in an attempt to classify them using a 4-linear layer model. Below, I have shared the my confusion matrix, showing the class predictions vs actual results, and a basic code of my linear model. 

{{< highlight python >}}
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(93, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, 10)
    
    def forward(self, x):
        x = x.view(-1, 92) 
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)
    
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
{{< / highlight >}}

![Confusion Matrix](/200430_otto/confusion_matrix.png)


To read more, please visit [my repository for this project](https://github.com/phillipluong/PyTorchProjects/tree/master/Otto%20Model)