from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_db = datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]), download=True)

test_db = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]),download=True)

def get_loader(batch_size):
    global train_db, test_db
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, _ = get_loader(32)
    x, label = next(iter(train_loader))
    print("x: ", x.shape, "\nlabel: ", label.shape)
