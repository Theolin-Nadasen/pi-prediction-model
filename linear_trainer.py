import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn

# first 100 digits of pi

HUNDRED_PI = np.array( [1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0,2,8,8,4,1,9,7,1,6,9,3,9,9,3,7,5,1,0,5,8,2,0,9,7,4,9,4,4,5,9,2,3,0,7,8,1,6,4,0,6,2,8,6,2,0,8,9,9,8,6,2,8,0,3,4,8,2,5,3,4,2,1,1,7,0,6,7,9] )
x = np.array( [x for x in range(1, 101)] )


# Visualize our data

#plt.scatter(x, HUNDRED_PI, c="r")
#plt.plot(x, HUNDRED_PI)
#plt.show()

# [ [1], [4], [1] ......]

x = torch.from_numpy(x).float()
y = torch.from_numpy(HUNDRED_PI).float()

x = torch.div(x , torch.tensor(100))
y = torch.div(y, torch.tensor(100))

x = x.unsqueeze(dim=1)
y = y.unsqueeze(dim=1)

# model

class basic_network(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer1(x)

model = basic_network()

# loss = abs(prediction - expected)

loss_fn = nn.L1Loss()


# optimizer

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# put the model in training mode
# get a prediction
# calculate the loss of that prediction
# zero the optimizer
# perform backpropagation and gradient descent

for i in range(5000):
    model.train()

    pred = model(x)

    loss = loss_fn(pred, y)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print(f"loss {loss}")

torch.save(model.state_dict(), "Linear_model.pth")