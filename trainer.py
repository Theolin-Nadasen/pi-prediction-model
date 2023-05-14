import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn

# first 100 digits of pi

HUNDRED_PI = np.array( [1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0,2,8,8,4,1,9,7,1,6,9,3,9,9,3,7,5,1,0,5,8,2,0,9,7,4,9,4,4,5,9,2,3,0,7,8,1,6,4,0,6,2,8,6,2,0,8,9,9,8,6,2,8,0,3,4,8,2,5,3,4,2,1,1,7,0,6,7,9] )
x = np.array( [x for x in range(1, 101)] )


# Visualize our data

plt.scatter(x, HUNDRED_PI, c="r")
plt.plot(x, HUNDRED_PI)
#plt.show()

x = torch.from_numpy(x).float().unsqueeze(dim=1)
x = torch.div(x, torch.tensor(100))

y = torch.from_numpy(HUNDRED_PI).float().unsqueeze(dim=1)
y = torch.div(y, torch.tensor(100))

#print(y)

# class

class model1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.layers(x)

model = model1()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# loop

epochs = 10

for epoch in range(epochs):
    model.train()

    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"loss {loss:.3f}")

with torch.inference_mode():
    model.eval()
    output = model( torch.tensor( [0.01, 0.02, 0.03, 0.04, 0.05]).float().unsqueeze(dim=1) )

print(output)
