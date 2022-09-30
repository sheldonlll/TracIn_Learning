from json import load
from statistics import mode
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from LeNet5 import LeNet5
from torch import optim

def load_data(test_batch_size = 32, train_batch_size = 32, download = False, shuffle = True, data_transform = {
        "train": transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.Resize((32, 32)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }):
    
    cifar10_train = datasets.CIFAR10(root = 'datasets', train = True, download = download, transform = data_transform["train"])
    cifar10_test = datasets.CIFAR10(root = 'datasets', train = False, download = download, transform = data_transform["val"])
    # cifar10_train size: 50000        cifar10_test_size: 10000 
    print(f"cifar10_train size: {len(cifar10_train)} \t cifar10_test_size: {len(cifar10_test)}")
    cifar10_train_dataloader = torch.utils.data.DataLoader(cifar10_train, batch_size = train_batch_size, shuffle = shuffle, num_workers = 0)
    cifar10_test_dataloader = torch.utils.data.DataLoader(cifar10_test, batch_size = test_batch_size, shuffle = shuffle, num_workers = 0)

    x, label = iter(cifar10_train_dataloader).next()
    test_x, test_label = iter(cifar10_test_dataloader).next()

    return x, label, test_x, test_label, cifar10_train_dataloader, cifar10_test_dataloader


def train_predict(cifar10_train, cifar10_test, epoches):
    device = torch.device("cuda")
    model = LeNet5().to(device)

    criteon = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in range(epoches):
        model.train()
        for batch_idx, (x, label) in enumerate(cifar10_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch}, loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            tot_correct = 0
            tot_num = 0
            for x, label in cifar10_test:
                x, label = x.to(device), label.to(device)
                logits = model(x) # [b, 10]
                pred = logits.argmax(dim = 1)

                tot_correct += torch.eq(pred, label).float().sum().item()
                tot_num += x.shape[0]
            acc = tot_correct / tot_num
            print(f"epoch: {epoch}, accuracy: {acc}")

def main():
    x, label, test_x, test_label, cifar10_train_dataloader, cifar10_test_dataloader = load_data()
#   cifar10_train size: 50000        cifar10_test_size: 10000 
#   torch.Size([32, 3, 32, 32])   torch.Size([32])   torch.Size([32, 3, 32, 32])   torch.Size([32])
    print(f"train shape: {x.shape} train label shape: {label.shape} test shape: {test_x.shape} test label shape: {test_label.shape}")
    train_predict(cifar10_train_dataloader, cifar10_test_dataloader, 1000)



if __name__ == "__main__":
    main()