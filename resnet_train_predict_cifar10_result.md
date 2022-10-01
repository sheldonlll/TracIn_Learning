# resnet34-cifar10结果

[resnet_cifar10_loss_accuracy_epoch_line](https://api2.mubu.com/v3/document_image/4f838be9-b944-4ce9-8f63-079aea4909d9-21226726.jpg)
enviroment, hyperparameters:
GPU: NVIDIA A40  CPU: 12 × Intel(R) Xeon(R) Gold 5318Y CPU @ 2.10GHz  内存: 86GB 硬盘: 350GB
test_batch_size = 256, train_batch_size = 1280,epoches = 25
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.ToTensor()
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
output:
Files already downloaded and verified
Files already downloaded and verified
    cifar10_train size: 50000      cifar10_test_size: 10000

train shape: torch.Size([1280, 3, 224, 224]) train label shape: torch.Size([1280]) test shape: torch.Size([256, 3, 224, 224]) test label shape: torch.Size([256])
epoch: 1, current epoch min loss: 1.4769628047943115
epoch: 1, accuracy: 0.3507
epoch: 2, current epoch min loss: 1.1555131673812866
epoch: 2, accuracy: 0.4587
epoch: 3, current epoch min loss: 0.9124709367752075
epoch: 3, accuracy: 0.5473
epoch: 4, current epoch min loss: 0.7965725660324097
epoch: 4, accuracy: 0.5424
epoch: 5, current epoch min loss: 0.6539226770401001
epoch: 5, accuracy: 0.6482
epoch: 6, current epoch min loss: 0.5314575433731079
epoch: 6, accuracy: 0.7026
epoch: 7, current epoch min loss: 0.47418370842933655
epoch: 7, accuracy: 0.7147
epoch: 8, current epoch min loss: 0.3847983479499817
epoch: 8, accuracy: 0.6632
epoch: 9, current epoch min loss: 0.3222113847732544
epoch: 9, accuracy: 0.726
epoch: 10, current epoch min loss: 0.2987883985042572
epoch: 10, accuracy: 0.791
epoch: 11, current epoch min loss: 0.2234237939119339
epoch: 11, accuracy: 0.7081
epoch: 12, current epoch min loss: 0.21544614434242249
epoch: 12, accuracy: 0.7946
epoch: 13, current epoch min loss: 0.17288713157176971
epoch: 13, accuracy: 0.7042
epoch: 14, current epoch min loss: 0.109545037150383
epoch: 14, accuracy: 0.7725
epoch: 15, current epoch min loss: 0.09756408631801605
epoch: 15, accuracy: 0.795
epoch: 16, current epoch min loss: 0.11165265738964081
epoch: 16, accuracy: 0.7371
epoch: 17, current epoch min loss: 0.07507659494876862
epoch: 17, accuracy: 0.7656
epoch: 18, current epoch min loss: 0.05396381765604019
epoch: 18, accuracy: 0.8091
epoch: 19, current epoch min loss: 0.047468364238739014
epoch: 19, accuracy: 0.8139
epoch: 20, current epoch min loss: 0.012718533165752888
epoch: 20, accuracy: 0.8287
epoch: 21, current epoch min loss: 0.010760265402495861
epoch: 21, accuracy: 0.8367
epoch: 22, current epoch min loss: 0.00593204889446497
epoch: 22, accuracy: 0.8497
epoch: 23, current epoch min loss: 0.006378650665283203
epoch: 23, accuracy: 0.8437
epoch: 24, current epoch min loss: 0.0032134423963725567
epoch: 24, accuracy: 0.8516
epoch: 25, current epoch min loss: 0.0017822051886469126
epoch: 25, accuracy: 0.8588