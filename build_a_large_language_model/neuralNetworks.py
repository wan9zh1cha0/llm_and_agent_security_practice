import torch
from torch.utils.data import Dataset
from torch.utils. data import DataLoader
import torch.nn.functional as F

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30), # number of nodes of input and output of this layer
            torch.nn.ReLU(), # linear layer + activation function

            # 2nd hidden layer
            torch.nn.Linear(30, 20), 
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

torch.manual_seed(123)
model = NeuralNetwork(50, 3)
torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)
print(out)

# NN for classification, input & target dataset
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

torch.manual_seed(123)
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}: {x}, {y}")

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5) # use stochastic gradient descent as optimizer
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d} | Batch {batch_idx:03d}/{len(train_loader):03d} | Train Loss: {loss:.2f}")
    model.eval()

model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)

#torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)
predictions = torch.argmax(probas, dim=1)
print(predictions)

torch.save(model.state_dict(), "model.pth")
model = NeuralNetwork(2, 2) 
model.load_state_dict(torch.load("model.pth"))