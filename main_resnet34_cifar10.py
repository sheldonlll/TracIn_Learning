import datetime
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torch import _pin_memory, optim
import os

from urllib3 import Retry

from mymodule.model import resnet34
from mymodule.pif.influence_functions_new import get_gradient, tracin_get

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def load_data(test_batch_size = 32, train_batch_size = 32, download = False, shuffle = True, ret_custom_all_data = False, category_num = 0, data_transform = {
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
    kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
    cifar10_train_dataloader = torch.utils.data.DataLoader(cifar10_train, batch_size = train_batch_size, shuffle = shuffle, num_workers = kwargs["num_workers"], pin_memory = kwargs["pin_memory"])
    cifar10_test_dataloader = torch.utils.data.DataLoader(cifar10_test, batch_size = test_batch_size, shuffle = shuffle, num_workers = kwargs["num_workers"], pin_memory = kwargs["pin_memory"])

    if ret_custom_all_data == False:
        x, label = iter(cifar10_train_dataloader).next()
        test_x, test_label = iter(cifar10_test_dataloader).next()
        return x, label, test_x, test_label, cifar10_train_dataloader, cifar10_test_dataloader

    else:
        train_data_iter = iter(cifar10_train_dataloader)
        train_image, train_label = train_data_iter.next()

        img_all_train = torch.zeros(500, 3, 32, 32) # 存放train_image中所有标签是参数category_num的图片作为训练数据集
        train_image_num = 0 # img_all_train数组的当前数量/下标，最多500张

        for i in range(train_batch_size):
            if (train_label[i] == category_num):
                img_all_train[train_image_num] = train_image[i]
                train_image_num += 1
            if train_image_num == 500:
                break
        


        test_data_iter = iter(cifar10_test_dataloader)
        test_image, test_label = test_data_iter.next()

        img_all_test = torch.zeros(100, 3, 32, 32) # 存放test_image中所有标签是参数category_num的图片作为训练数据集
        test_image_num = 0 # img_all_train数组的当前数量/下标，最多100张

        for i in range(test_batch_size):
            if (test_label[i] == category_num):
                img_all_test[test_image_num] = test_image[i]
                test_image_num += 1
            if test_image_num == 100:
                break
        
        return img_all_train, img_all_test # shape: (500, 3, 224, 224), (100, 3, 224, 224)



def train_predict_save_per_epoch(cifar10_train, cifar10_test, epoches, checkpoint_path = "./resnet_cifar10_cpts/"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet34().to(device)

    criteon = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in range(epoches):
        model.train()
        loss = torch.tensor(-1.0)
        lossMIN = 0x3fff
        launchTimestamp = datetime.datetime.now()
        for _batch_idx, (x, label) in enumerate(cifar10_train):
            x, label = x.to(device), label.to(device)
            try:
                logits = model(x)
                loss = criteon(logits, label)
                lossMIN = min(lossMIN, loss)
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            

        print(f"epoch: {epoch + 1}, loss: {loss.item()}")

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
            print(f"epoch: {epoch + 1}, accuracy: {acc}")
        
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()},
                           checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')

    return model

def calculate_influence(img_all_train, img_all_test, net):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.CrossEntropyLoss()
    img_all_train = img_all_train.view(500, 1, 3, 224, 224)
    img_all_test = img_all_test.view(100, 1, 3, 224, 224)
    label_train = torch.zeros(1).long()
    logits_train = net(img_all_train[0].to(device))
    loss_train = loss_fn(logits_train, label_train.to(device))
    grad_z_train = torch.autograd.grad(loss_train, net.parameters())
    grad_z_train = get_gradient(grad_z_train, net)
    score_list = []
    import time
    time_start = time.perf_counter()
    for i in range(10000):
        label_test = torch.zeros(1).long()
        label_test[0] = 0
        logits_test = net(img_all_test[i].to(device))
        loss_test = loss_fn(logits_test, label_test.to(device))
        grad_z_test = torch.autograd.grad(loss_test, net.parameters())
        grad_z_test = get_gradient(grad_z_test, net)
        score = tracin_get(grad_z_test, grad_z_train)
        score_list.append(float(score))
        print(score)
    print('%f s' % (time.perf_counter() - time_start))
    print(score_list)


def main():
    torch.cuda.empty_cache()
    #   cifar10_train size: 50000        cifar10_test_size: 10000 
    #   torch.Size([32, 3, 32, 32])   torch.Size([32])   torch.Size([32, 3, 32, 32])   torch.Size([32])
    x, label, test_x, test_label, cifar10_train_dataloader, cifar10_test_dataloader = load_data(test_batch_size = 600, train_batch_size = 3000, category_num = 0, ret_custom_all_data=False)
   
    print(f"train shape: {x.shape} train label shape: {label.shape} test shape: {test_x.shape} test label shape: {test_label.shape}")
    net = train_predict_save_per_epoch(cifar10_train_dataloader, cifar10_test_dataloader, 1000, checkpoint_path="./resnet_cifar10_cpts/")

    img_all_train, img_all_test = load_data(test_batch_size = 600, train_batch_size = 3000, category_num = 0, ret_custom_all_data = True)
    calculate_influence(img_all_train, img_all_test, net)
    

    '''
    img_all_train, img_all_test = load_data(test_batch_size = 1000, train_batch_size = 5000, category_num = 0)
    train_predict(img_all_train, img_all_test, 1000)
    '''

if __name__ == "__main__":
    main()