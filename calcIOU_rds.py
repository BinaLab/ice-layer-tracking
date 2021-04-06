# debvrat
# from __future__ import print_function
import os
import sys
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

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

num_classes = 28
gt_root = 'G:/My Drive/Debvrat_Research/Dataset/RDS/'
out_root = 'G:/My Drive/Debvrat_Research/Outputs4/RDS/SemanticSegmentation/'
dataset = 'MacGregor_200721'
# rds_main = ['MacGregor_200721','MacGregor_200824/frames_001_016_20120507_05',
# 'MacGregor_200824/frames_001_016_20120508_06', 'MacGregor_200824/frames_001_018_20120508_04',
# 'MacGregor_200824/frames_001_028_20120507_07','MacGregor_200824/frames_001_030_20120330_03']

# val
rds_main = ['MacGregor_200824/frames_001_016_20120508_06'] 
# train
rds_main = ['MacGregor_200824/frames_001_016_20120507_05', 'MacGregor_200824/frames_001_018_20120508_04',
'MacGregor_200824/frames_001_028_20120507_07','MacGregor_200824/frames_001_030_20120330_03']


# models = ['UNet/08-18_13-44','UNet/08-19_12-24', 'PSPNet/08-18_17-54', 'PSPNet/08-18_23-40', 
#           'DeepLab/08-14_11-02', 'DeepLab/08-14_16-52']
# models = ['DeepLab/08-14_11-02']
# model = 'DeepLab/08-14_11-02'
model = 'DeepLab/10-04_15-40'
layer_semantic = 'layer_semantic'
png_ext = '.png'
image = 'image'
layer = 'layer'
pred_arr = []
true_arr = []
f = open(os.path.join(out_root,model,'metrics_train.txt'), 'w+')
for dataset in rds_main:
    # for model in models:
    print(model)
    model_path = os.path.join(out_root,model,dataset)
    gt_path = os.path.join(gt_root,dataset,layer_semantic)
    # os.chdir(os.path.join(out_root,model))
    files = [el for el in os.listdir(model_path) if png_ext in el]
    # pred_arr = []
    # true_arr = []
    i = 0
    total = len(files)
    for file in files:
        i+=1
        # pred = np.array(Image.open(os.path.join(model_path, file))).flatten()
        pred = np.array(Image.open(os.path.join(model_path, file))).flatten()
        true = np.array(Image.open(os.path.join(gt_path, file.replace(image, layer))))[:,:,0].flatten()
        # true = np.array(Image.open(os.path.join(gt_path, file.replace(image, layer))))[:,:].flatten()
        pred_arr += pred.tolist()
        true_arr += true.tolist()
        print('\r   '+dataset+' ' +str(int(i*100/total))+'%', end='', flush=True)
    print('')
    # pixAcc, mIoU = get_seg_metrics(pred_arr, true_arr, num_classes)
    # f.write(dataset +' acc '+ str(pixAcc) + ' mIoU ' + str(mIoU) + '\n')
averaging = 'micro'
pixAcc, mIoU = get_seg_metrics(pred_arr, true_arr, num_classes)
p, r, f1, _ = precision_recall_fscore_support(true_arr,pred_arr,average=averaging)
f.write('acc '+ str(pixAcc) + ' mIoU ' + str(mIoU) + '\n')
f.write('prec ' + str(p) + ' rec ' + str(r) + ' f-score ' + str(f1) + ' [' +averaging + ']')
f.close()
print('done')