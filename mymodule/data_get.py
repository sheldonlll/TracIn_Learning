import torchvision.transforms as transforms
import torchvision
import torch
def dataset_get():
    pass

def dataset_category_get(category_num):
    '''
    transforms.ToTensor():
    converts images loaded by Pillow into PyTorch tensors.

    transforms.Normalize():
    adjusts the values of the tensor
    so that their average is [0.485, 0.456, 0.406] and their standard deviation is [0.229, 0.224, 0.225].
    Most activation functions have their strongest gradients around x = 0, so centering our data there can speed learning.
    There are many more transforms available, including cropping, centering, rotation, and reflection.
    '''
    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224), #TODO
                                     transforms.ToTensor(),# converts images loaded by Pillow into PyTorch tensors.
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    } # transform incoming images into a Pytorch Tensor
    
    '''
    CIFAR10
    This is a set of 32x32 color image tiles representing 10 classes of objects: 
    6 of animals (bird, cat, deer, dog, frog, horse) and 4 of vehicles (airplane, automobile, ship, truck)
    
    The Dataset retrieves our dataset’s features and labels one sample at a time. 
    While training a model, we typically want to pass samples in “minibatches”, 
    reshuffle the data at every epoch to reduce model overfitting, 
    and use Python’s multiprocessing to speed up data retrieval.

    DataLoader is an iterable that abstracts this complexity for us in an easy API.
    '''
    
    test_set = torchvision.datasets.CIFAR10(root = "datasets", train = False, download = False, transform = data_transform["val"])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 2000, shuffle = False, num_workers = 0)

    train_set = torchvision.datasets.CIFAR10(root = "datasets", train = False, download = False, transform = data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 10000, shuffle = False, num_workers = 0)
    
    '''
    We have loaded that dataset into the DataLoader and can iterate through the dataset as needed. 
    Each iteration below returns a batch of train_features and train_labels (containing batch_size=64 features and labels respectively). 
    Because we specified shuffle=False, after we iterate over all batches the data is not shuffled.
    '''

    train_data_iter = iter(train_loader)
    train_image, train_label = train_data_iter.next()

    img_all_train = torch.zeros(500, 3, 224, 224) # 存放train_image中所有标签是参数category_num的图片作为训练数据集
    train_image_num = 0 # img_all_train数组的当前数量/下标，最多500张

    for i in range(10000):
        if (train_label[i] == category_num):
            img_all_train[train_image_num] = train_image[i]
            train_image_num += 1
        if train_image_num == 500:
            break
    


    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()

    img_all_test = torch.zeros(100, 3, 224, 224) # 存放test_image中所有标签是参数category_num的图片作为训练数据集
    test_image_num = 0 # img_all_train数组的当前数量/下标，最多100张

    for i in range(2000):
        if (test_label[i] == category_num):
            img_all_test[test_image_num] = test_image[i]
            test_image_num += 1
        if test_image_num == 100:
            break
    
    return img_all_train, img_all_test # shape: (500, 3, 224, 224), (100, 3, 224, 224)
    
    
