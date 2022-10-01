import os
import time
import torch
from torch import nn
from mymodule.pif.influence_functions_new import get_gradient, tracin_get
from mymodule.model import resnet34
from mymodule.data_get import dataset_category_get

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model_weight_path = "./resnet_cifar10_cpts/resNet34_epoch1.pth.tar"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    
    net = resnet34()
    checkpoint = torch.load(model_weight_path, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    print(f"")
    loss_fn =  nn.CrossEntropyLoss()
    
    img_all_train,img_all_test = dataset_category_get(0) # 取第0种图片
    train_size = 500
    test_size = 100
    img_all_train = img_all_train.view(train_size, 1, 3, 224, 224)
    img_all_test = img_all_test.view(test_size, 1, 3, 224, 224)

    label_train = torch.zeros(1).long()#0是真实标签
    logits_train = net(img_all_train[0].to(device))
    loss_train = loss_fn(logits_train, label_train.to(device))
    grad_z_train = torch.autograd.grad(loss_train, net.parameters())
    grad_z_train = get_gradient(grad_z_train, net)

    score_list = []
    time_start = time.perf_counter()
    
    for i in range(test_size):
        label_test = torch.zeros(1).long()
        label_test[0] = 0
        logits_test = net(img_all_test[i].to(device))
        loss_test = loss_fn(logits_test, label_test.to(device))
        grad_z_test = torch.autograd.grad(loss_test, net.parameters())
        grad_z_test = get_gradient(grad_z_test, net)

        score = tracin_get(grad_z_test, grad_z_train)
        score_list.append(float(score))

    print('%f s' % (time.perf_counter() - time_start))
    print(score_list)
'''
Files already downloaded and verified
2.731758 s
[743.8076782226562, 733.111572265625, 752.1342163085938, 721.2263793945312, 753.67041015625, 717.1847534179688, 750.5286865234375, 742.0112915039062, 754.403076171875, 700.0720825195312, 752.8794555664062, 714.9146118164062, 752.3242797851562, 746.2551879882812, 719.6704711914062, 765.6649169921875, 747.4202880859375, 728.2677001953125, 720.2879028320312, 720.0881958007812, 744.110107421875, 735.44091796875, 721.948974609375, 766.2697143554688, 749.6228637695312, 745.3672485351562, 748.4539184570312, 757.8270263671875, 743.2897338867188, 732.8866577148438, 763.4796142578125, 732.037109375, 765.2484741210938, 747.7365112304688, 767.2817993164062, 749.83203125, 754.1848754882812, 762.0814208984375, 734.998291015625, 749.23681640625, 739.7030639648438, 744.4843139648438, 734.5945434570312, 758.3264770507812, 765.2114868164062, 717.6148681640625, 743.0769653320312, 748.2489013671875, 730.869384765625, 744.3200073242188, 761.2022094726562, 729.6329956054688, 744.208984375, 730.6611328125, 748.0765380859375, 734.6636962890625, 749.200927734375, 746.7312622070312, 757.3738403320312, 745.300537109375, 749.7775268554688, 747.8408203125, 754.1475830078125, 715.4417724609375, 731.6383666992188, 731.5656127929688, 742.351806640625, 743.7654418945312, 745.63525390625, 771.5394897460938, 728.6211547851562, 718.993896484375, 735.7783813476562, 739.1575927734375, 761.4649047851562, 741.9418334960938, 746.58837890625, 737.979736328125, 756.896240234375, 744.858154296875, 743.561767578125, 764.5882568359375, 754.668212890625, 764.8370361328125, 749.6792602539062, 746.0980224609375, 765.5466918945312, 775.1990966796875, 747.3214721679688, 758.5355834960938, 742.328125, 702.4049682617188, 745.525146484375, 729.591796875, 739.27490234375, 774.824951171875, 746.283935546875, 731.7025146484375, 749.4126586914062, 703.1405029296875]
'''
main()

