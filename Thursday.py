import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X = torch.randn(1000, 2)

y = (X[:, 0]+ X[:, 1] > 0).long()

print(X.shape)
print(y.shape)
print(y[:10])

class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,2)
    def forward(self,x):
        return self.linear(x)

model = LinearClassifier()
print(model)

criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr = 0.1)

print(criterion)
print(optimiser)