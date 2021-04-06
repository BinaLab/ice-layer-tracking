import argparse
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
import re # @dv
import matplotlib.pyplot as plt # @dv
import cv2 # @dv

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

def sliding_predict(model, image, num_classes, flip=True):
    image_size = image.shape
    tile_size = (int(image_size[2]//2.5), int(image_size[3]//2.5))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = padded_img.flip(-1)
                fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions
    return total_predictions


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu()) # original, commented by @dv
        # model_out = model(scaled_img) # @dv: for dsn
        # model_mean = np.mean(np.dstack((model_out[0].cpu(),model_out[1].cpu(),model_out[2].cpu())),axis=2) # @dv: for dsn
        # scaled_prediction = upsample(model_mean) # @dv: for dsn

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions

def final_classification_dv(prediction):
    d, h, w = prediction.shape
    pred = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            depth_vector = prediction[:,i,j]
            idx = np.argmax(depth_vector)
            pred[i,j] = idx
    return pred
            


def save_images(image, mask, output_path, image_file, palette):
	# Saves the image, the model output and the results after the post processing
    w, h = image.size # original, for PIL probably @dv
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file+'.png'))
    # output_im = Image.new('RGB', (w*2, h))
    # output_im.paste(image, (0,0))
    # output_im.paste(colorized_mask, (w,0))
    # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))

def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K', 'CReSIS', 'RDS']
    if dataset_type == 'CityScapes': 
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] 
    else:
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel): # commented by @dv
        model = torch.nn.DataParallel(model)         # commented by @dv

    model.load_state_dict(checkpoint) # original @dv
    model.to(device)
    model.eval()

    network_timestamp = re.search('/(.+?)/checkpoint-epoch70', args.model) # @dv
    # network_timestamp = re.search('/(.+?)/best_model', args.model) # @dv
   
    # args.output = os.path.join(args.output, network_timestamp.group(1), args.data_type) # @dv: was using this earlier
   
    # root = 'G:/My Drive/Debvrat_Research/Dataset/Snow Radar/'
    root = 'G:/My Drive/Debvrat_Research/Dataset/Snow Radar/final_on_AWS/' # @dv: windows
    # root = '/Volumes/GoogleDrive/My Drive/Debvrat_Research/Dataset/RDS/' # @dv: mac
    
    # rds_types = ['2009_dv', '2010_dv', '2011_dv', '2013_dv',
    #              '2014_dv', '2015_dv', '2016_dv', '2017_dv']
    
    # rds_types = ['Corrected_SEGL_picks_lnm_2009_2017_reformat/2013','Corrected_SEGL_picks_lnm_2009_2017_reformat/2014',
    #              'Corrected_SEGL_picks_lnm_2009_2017_reformat/2015','Corrected_SEGL_picks_lnm_2009_2017_reformat/2016',
    #              'Corrected_SEGL_picks_lnm_2009_2017_reformat/2017','greenland_picks_final_2009_2012_reformat/2009',
    #              'greenland_picks_final_2009_2012_reformat/2010','greenland_picks_final_2009_2012_reformat/2011',
    #              'greenland_picks_final_2009_2012_reformat/2012']
    rds_types = ['Corrected_SEGL_picks_lnm_2009_2017_reformat/2009','Corrected_SEGL_picks_lnm_2009_2017_reformat/2010',
                 'Corrected_SEGL_picks_lnm_2009_2017_reformat/2011','Corrected_SEGL_picks_lnm_2009_2017_reformat/2012',
                 'Corrected_SEGL_picks_lnm_2009_2017_reformat/2013','Corrected_SEGL_picks_lnm_2009_2017_reformat/2014',
                 'Corrected_SEGL_picks_lnm_2009_2017_reformat/2015','Corrected_SEGL_picks_lnm_2009_2017_reformat/2016',
                 'Corrected_SEGL_picks_lnm_2009_2017_reformat/2017',
                 'greenland_picks_final_2009_2012_reformat/2009',
                 'greenland_picks_final_2009_2012_reformat/2010','greenland_picks_final_2009_2012_reformat/2011',
                 'greenland_picks_final_2009_2012_reformat/2012']
    # rds_types = ['greenland_picks_final_2009_2012_reformat/2009',
    #              'greenland_picks_final_2009_2012_reformat/2010','greenland_picks_final_2009_2012_reformat/2011',
    #              'greenland_picks_final_2009_2012_reformat/2012']
    for rds in rds_types:
        
        root_path = os.path.join(root,rds)
        out_main = os.path.join(args.output, network_timestamp.group(1),rds)
        if not os.path.exists(out_main):
            os.makedirs(out_main)
        image_files = sorted(glob(os.path.join(root_path,'image_cropped', f'*.png')))
      
        with torch.no_grad():
            tbar = tqdm(image_files, ncols=100)
            for img_file in tbar:
                torch.cuda.empty_cache()
                # image = Image.open(img_file).convert('RGB') # original, commented @dv
                # @dv: next 3 lines for processing tiffs of RDS
                image = Image.open(img_file).convert('RGB') 
                if image is None:
                    print(img_file+' gives None')
                    continue
                print(img_file + ' ' + str(image.size))
                if ('SegNet' in  network_timestamp.group(1)) and (image.size[1] < 42): # @dv: SegNet check
                    continue
                input = normalize(to_tensor(image)).unsqueeze(0)
                
                if args.mode == 'multiscale':
                    prediction = multi_scale_predict(model, input, scales, num_classes, device)
                elif args.mode == 'sliding':
                    prediction = sliding_predict(model, input, num_classes)
                else:
                    prediction = model(input.to(device))
                    prediction = prediction.squeeze(0).cpu().numpy()
                # pred = final_classification_dv(prediction) # @dv
                # prediction = prediction[1:,:,:] # @dv
                prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
                save_images(image, prediction, out_main, img_file, palette) # color palette - commented by @dv
                # plt.imsave(os.path.join(out_main,os.path.basename(img_file.replace('.tiff','.png'))),prediction.astype(np.uint8)) # to generate viridis palette @dv
                # prediction = cv2.cvtColor(prediction,cv2.COLOR_GRAY2RGB) # @dv: prediction try
                # cv2.imwrite(os.path.join(out_main,os.path.basename(img_file.replace('.tiff','.png'))), prediction.astype(np.uint8)) # @dv: prediction try
                # cv2.imwrite(os.path.join(out_main,os.path.basename(img_file.replace('.tiff','.png'))), prediction) # @dv: prediction try

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config_snow_radar.json',type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='saved/DeepLab/11-02_17-23/checkpoint-epoch70.pth', type=str, # @dv: snow radar model path
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-d', '--dataset', default='2012_cropped', type=str, # @dv, added: 2012_main or 2012_cropped
                        help='which data to be segmented - 2012_main or 2012_cropped')
    parser.add_argument('-dt', '--data_type', default='train_split', type=str, # @dv, added: training set or test set of CReSIS
                        help='\'test\' or \'train\' or \'val\' or \'train_split\' of CReSIS which needs to be segmented')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='G:/My Drive/Debvrat_Research/Outputs/SnowRadar/SemanticSegmentation', type=str,  
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='png', type=str, # @dv: author's default = 'jpg'
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
