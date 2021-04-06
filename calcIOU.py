# debvrat
# from __future__ import print_function
import os
import sys
import numpy as np
from PIL import Image

def pixel_accuracy(output, target):
    output = np.asarray(output)
    target = np.asarray(target)
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((output == target) * (target > 0))
    pixel_correct = np.sum((output == target))
    return pixel_correct, pixel_labeled

def inter_over_union(output, target, num_class):
    output = np.asarray(output) + 1
    target = np.asarray(target) + 1
    output = output * (target > 0)

    intersection = output * (output == target)
    area_inter, _ = np.histogram(intersection, bins=num_class, range=(1, num_class))
    area_pred, _ = np.histogram(output, bins=num_class, range=(1, num_class))
    area_lab, _ = np.histogram(target, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union

def eval_metrics(output, target, num_classes):
    correct, labeled = pixel_accuracy(output, target)
    inter, union = inter_over_union(output, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]

def get_seg_metrics(output, target, num_classes):
    correct, labeled, inter, union = eval_metrics(output, target, num_classes)
    pixAcc = 1.0 * correct / (np.spacing(1) + labeled)
    IoU = 1.0 * inter / (np.spacing(1) + union)
    mIoU = IoU.mean()
    return np.round(pixAcc, 3), np.round(mIoU, 3)
    # return np.round(pixAcc, 3), np.round(mIoU, 3), dict(zip(range(self.num_classes), np.round(IoU, 3)))

num_classes = 27
num_classes = 20
# gt_root = 'G:/My Drive/Debvrat_Research/Dataset/2012_cropped_semantic/png'
gt_root = 'G:/My Drive/Debvrat_Research/Darwin_Backup/Arctic2012_test-final/semantic'
out_root = 'G:/My Drive/Debvrat_Research/Outputs/SnowRadar/SemanticSegmentation/'
# dataset = '2012_cropped'
dataset = '2012_main_dv'
# models = ['UNet/08-04_10-35','UNet/08-04_14-39', 'DeepLab/07-18_17-59', 'DeepLab/07-20_21-44', 
#           'PSPNet/07-17_03-37', 'PSPNet/07-21_02-09']

# models = ['UNet/08-18_13-44','UNet/08-19_12-24', 'PSPNet/08-18_17-54', 'PSPNet/08-18_23-40', 
#           'DeepLab/08-14_11-02', 'DeepLab/08-14_16-52']
models = ['UNet/08-18_13-44','UNet/08-19_12-24','PSPNet/08-18_17-54', 'PSPNet/08-18_23-40',
          'DeepLab/08-14_11-02', 'DeepLab/08-14_16-52']
f = open('C:/Users/debvrav1/Desktop/metrics5.txt','w+')
for model in models:
    print(model)
    model_path = os.path.join(out_root,model,dataset)
    folders = [el for el in os.listdir(model_path) if os.path.isdir(os.path.join(model_path,el))]
    os.chdir(model_path)
    # f = open('metrics2.txt', 'w+')
    for folder in folders:
        if folder!='test':  ### @dv: to calculate only over test set for now
            continue
        folder_path = os.path.join(model_path,folder)
        files = [el for el in os.listdir(folder_path) if '.png' in el]
        pred_arr = []
        true_arr = []
        i = 0
        total = len(files)
        print('total files = ' + str(total))
        j = 0 # index to calculate non single label images
        for file in files:
            i+=1
            pred = np.array(Image.open(os.path.join(folder_path, file))).flatten()
            # true = np.array(Image.open(os.path.join(gt_root, file.replace('image', 'layer')))).flatten()
            true = np.array(Image.open(os.path.join(gt_root, file))).flatten()
            if len(np.unique(true)) <= 2: ## skip files which have just a single label: [0,1]
                continue
            j+=1
            pred = np.where(pred<21,pred,0) ## calculate metrics on only the top 10 layers
            true = np.where(true<21, true,0) ## calculate metrics on only the top 10 layers
            pred_arr += pred.tolist()
            true_arr += true.tolist()
            print('\r   '+folder+' ' +str(int(i*100/total))+'%', end='', flush=True)
        print('Non-single label images = ' + str(j))
        # print('More than 3 labels images = ' + str(j))
        print('')
        pixAcc, mIoU = get_seg_metrics(pred_arr, true_arr, num_classes)
        f.write(model + ' ' + folder +' acc '+ str(pixAcc) + ' mIoU ' + str(mIoU) +'\n')
    # f.close()
    print('')

f.close()        
        
# 'DeepLab/07-20_21-44/2012_cropped/train_split'
# file_name = 'image_20120330_02_018__crop1.png'

# pred_root = 'G:/My Drive/Debvrat_Research/Outputs/SnowRadar/SemanticSegmentation/DeepLab/07-20_21-44/2012_cropped/train_split'



# pred = np.array(Image.open(os.path.join(pred_root, file_name))).flatten()
# true = np.array(Image.open(os.path.join(gt_root, file_name).replace('image', 'layer'))).flatten()

# pixAcc, mIoU = get_seg_metrics(pred, true, num_classes)
# print(pixAcc, mIoU)








### lr scheduler ###

# import torchvision
# import torch
# import matplotlib.pylab as plt
# from  torch.optim.lr_scheduler import _LRScheduler

# resnet = torchvision.models.resnet34()
# params = {
#     "lr": 0.01,
#     "weight_decay": 0.001,
#     "momentum": 0.9
# }
# optimizer = torch.optim.SGD(params=resnet.parameters(), **params)

# epochs = 5
# iters_per_epoch = 100
# lrs = []
# mementums = []
# # lr_scheduler = OneCycle(optimizer, epochs, iters_per_epoch)
# # lr_scheduler = Poly(optimizer, epochs, iters_per_epoch)

# for epoch in range(epochs):
#     for i in range(iters_per_epoch):
#         _LRScheduler.step(epoch=epoch)
#         _LRScheduler(optimizer, i, epoch)
#         lrs.append(optimizer.param_groups[0]['lr'])
#         # mementums.append(optimizer.param_groups[0]['momentum'])
        

# plt.ylabel("learning rate")
# plt.xlabel("iteration")
# plt.plot(lrs)
# plt.show()

# # plt.ylabel("momentum")
# # plt.xlabel("iteration")
# # plt.plot(mementums)
# # plt.show()
