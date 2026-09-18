from vgg_data import get_loader
from vgg_model import VGG13

import torch
from torch import nn, optim

device = torch.device("cuda")
model = VGG13().to(device)
train_loader, _ = get_loader(32)
criterion = nn.CrossEntropyLoss().to(device)
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=1e-4)
num_epoch=30

for epoch in range(num_epoch):
    model.train()
    epoch_loss = 0
    total_samples = 0
    for step, (x, label) in enumerate(train_loader):
        optimizer.zero_grad()
        x, label = x.to(device), label.to(device)
        logits = model(x)
        #logits: [batch_size, 10]
        loss = criterion(logits, label)
        loss.backward()

        batch_size = x.size(0)
        epoch_loss += loss.item() * batch_size
        total_samples += batch_size

        optimizer.step()
    print(f"Epoch: {epoch+1}/{num_epoch}, Loss: {epoch_loss/total_samples:.4f}")


torch.save(model.state_dict(), "weights.pth")


