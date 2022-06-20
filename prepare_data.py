"""
This script is used to create Voronoi labels and cluster labels from the point labels
and prepare the dataset for training and testing.

Author: Hui Qu
"""

import os
import shutil
import numpy as np
import torch.cuda
from skimage import morphology, measure, io
from sklearn.cluster import KMeans
from scipy.ndimage.morphology import distance_transform_edt as dist_tranform
import glob
import json

from options import Options
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.pyplot as plt
from read_roi import read_roi_file
from PIL import Image, ImageDraw
import random
import colorsys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(opt):
    # opt.ratio = 0.05
    dataset = opt.dataset
    ratio = opt.ratio
    nuclei_channel = opt.nuclei_channel
    nuclei_channel_image = opt.nuclei_channel_image
    bit_value = opt.bit
    marker_channel_image = opt.marker_channel_image
    marker_channel = opt.marker_channel
    data_dir = '../data/{:s}'.format(dataset)
    img_dir = '../Image_classification/{:s}/TIFF_wsi'.format(dataset)
    img_dir_rgb = '../data/{:s}/images_RGB'.format(dataset)
    img_dir_class = '../data/{:s}/classification_labels'.format(dataset)
    label_point_dir_2D = '../Image_classification/{:s}/labels_point/2D/'.format(dataset)
    label_point_dir_3D = '../Image_classification/{:s}/labels_point/3D/'.format(dataset)
    rois_path = '../Image_classification/{:s}/LesionROI/'.format(dataset)
    ROI_nuclei_dir = '../Image_classification/{:s}/ROI_nuclei/'.format(dataset)
    gt_class_label = '../Image_classification/{:s}/gt_class_label/'.format(dataset)

    label_vor_dir = '../Image_classification/{:s}/labels_voronoi_{:.2f}'.format(dataset, ratio)
    label_cluster_dir = '../Image_classification/{:s}/labels_cluster_{:.2f}'.format(dataset, ratio)
    Marker_range_dir = '../Image_classification/{:s}/Marker_{:.2f}'.format(dataset, ratio)
    rgb_path = '../Image_classification/{:s}/RGB_maker_nuclei'.format(dataset)
    patch_folder = '../data/{:s}/patches'.format(dataset)
    train_data_dir = '../data_for_train/{:s}'.format(dataset)

    with open('../Image_classification/{:s}/train_val_test.json'.format(dataset), 'r') as file:
        data_list = json.load(file)
        train_list = data_list['train']

    # ------ create point label
#     nuclei_point(img_dir, label_point_dir_2D, label_point_dir_3D, dataset, train_list, nuclei_channel_image, rois_path, ROI_nuclei_dir)
#     nuclei_point(img_dir, label_point_dir_2D, label_point_dir_3D, dataset, data_list['val'], nuclei_channel_image, rois_path, ROI_nuclei_dir)
#     nuclei_point(img_dir, label_point_dir_2D, label_point_dir_3D, dataset, data_list['test'], nuclei_channel_image, rois_path, ROI_nuclei_dir)



    # ------ create Voronoi label from point label
    create_Voronoi_label(label_point_dir_2D, label_vor_dir, train_list, nuclei_label=nuclei_channel) 
    create_Voronoi_label(label_point_dir_2D, label_vor_dir, data_list['val'], nuclei_label=nuclei_channel) 
    create_Voronoi_label(label_point_dir_2D, label_vor_dir, data_list['test'], nuclei_label=nuclei_channel)

    
    
    # ------ create cluster label from point label and image
    create_cluster_label(ROI_nuclei_dir, label_point_dir_3D, label_vor_dir, label_cluster_dir, train_list, nuclei_channel_image = nuclei_channel_image, bit=bit_value, img_dir=img_dir, marker_channel_image=marker_channel_image, Marker_range_dir=Marker_range_dir, cluster_label = True, rois_path_=rois_path)

    create_cluster_label(ROI_nuclei_dir, label_point_dir_3D, label_vor_dir, label_cluster_dir, data_list['val'], nuclei_channel_image = nuclei_channel_image, bit=bit_value, img_dir=img_dir, marker_channel_image=marker_channel_image, Marker_range_dir=Marker_range_dir, cluster_label = True, rois_path_=rois_path)
    
    create_cluster_label(ROI_nuclei_dir, label_point_dir_3D, label_vor_dir, label_cluster_dir, data_list['test'], nuclei_channel_image = nuclei_channel_image, bit=bit_value, img_dir=img_dir, marker_channel_image=marker_channel_image, Marker_range_dir=Marker_range_dir, cluster_label = True, rois_path_=rois_path)



    #--------Generating Instance labels
    instance_labels(label_cluster_dir, dataset, train_list,n_channel=4, gt_class_label=gt_class_label)
    instance_labels(label_cluster_dir, dataset, data_list['val'],n_channel=4, gt_class_label=gt_class_label)
    
    instance_labels(label_cluster_dir, dataset, data_list['test'],n_channel=4, gt_class_label=gt_class_label)

    
    
    
    #---------Multichannel (Nuclei+marker)
    multichannel(rgb_path, train_list,ROI_nuclei_dir, Marker_range_dir)
    multichannel(rgb_path, data_list['val'],ROI_nuclei_dir, Marker_range_dir)
    multichannel(rgb_path, data_list['test'],ROI_nuclei_dir, Marker_range_dir)
        

    # ------ split large images into 250x250 patches
    if os.path.exists(patch_folder):
        shutil.rmtree(patch_folder)
    create_folder(patch_folder)
    print("Spliting large images into small patches...")
    split_patches(rgb_path, '{:s}/images'.format(patch_folder))
    split_patches(label_vor_dir, '{:s}/labels_voronoi'.format(patch_folder), 'label_vor')
    split_patches(label_cluster_dir, '{:s}/labels_cluster'.format(patch_folder), 'label_cluster')
    split_patches(label_point_dir_2D, '{:s}/labels_point'.format(patch_folder), 'label_point')
    split_patches(gt_class_label, '{:s}/classification_labels'.format(patch_folder))

    # ------ divide dataset into train, val and test sets
    organize_data_for_training(data_dir, train_data_dir, dataset)

    # ------ compute mean and std
    compute_mean_std(rgb_path, train_data_dir, dataset)

    # ------ Hovernet_data
    hovernet_format(data = 'train', train_data_dir = train_data_dir, marker_channel=marker_channel)
    hovernet_format(data = 'test', train_data_dir = train_data_dir, marker_channel=marker_channel)
  
    
    
    
    
    
def multichannel(rgb_path, train_list,ROI_nuclei_dir, Marker_range_dir):
    
    create_folder(rgb_path)
    for img_name in train_list:
        if img_name.endswith('png'):
            name = img_name.split('.')[0]
            nuclei = np.load(os.path.join(ROI_nuclei_dir, name + '.npy'))
            nuclei = bytescale(nuclei)
            nuclei = np.max(nuclei, axis=2)
    
            marker = np.load(os.path.join(Marker_range_dir, name + '.npy'))
            # marker = bytescale(marker)
            marker = np.max(marker, axis=2)
            channel1 = marker
            channel2 = np.zeros((nuclei.shape[0], nuclei.shape[1]))
            channel3 = nuclei
            channel4 = np.zeros((nuclei.shape[0], nuclei.shape[1]))
    
            img = np.concatenate(
                [channel1[np.newaxis, :], channel2[np.newaxis, :], channel3[np.newaxis, :], channel4[np.newaxis, :]])
            img = np.moveaxis(img, 0, 2)
            n_channels = img.shape[2]
            colors = np.array(generate_colors(n_channels))
            out_shape = list(img.shape)
            out_shape[2] = 3  ## change to RGB number of channels (3)
            out = np.zeros(out_shape)
            for chan in range(img.shape[2]):
                out = out + np.expand_dims(img[:, :, chan], axis=2) * np.expand_dims(colors[chan] / 255, axis=0)
            out = out / np.max(out)
            out = Image.fromarray((out * 255).astype(np.uint8))
    
            out.save(os.path.join(rgb_path, name)+ '.png')

        
def hovernet_format(data, train_data_dir, marker_channel):   
    img_path = '{:s}/images/{:s}'.format(train_data_dir,data)
    print(img_path)
    label_path = '{:s}/labels_cluster/{:s}'.format(train_data_dir,data)
    class_path = '{:s}/classification_labels/{:s}'.format(train_data_dir,data)
    path_save = '{:s}/Hovernet_data/{:s}/'.format(train_data_dir,data)
    create_folder(path_save)
    list_img=os.listdir(img_path)
    list_label=os.listdir(label_path)
    list_class=os.listdir(class_path)
    n=0
    for img in list_img:
      if img.endswith('png'):
#         print(img)
        img_rgb=np.array(Image.open(os.path.join(img_path,img)).convert('RGB'))
        label_id=img[:-4]+'_label_cluster.png'
        a=np.array(Image.open(os.path.join(label_path,label_id)))
        b=a[:,:,1]
        c=a[:,:,0]
        semi=np.zeros((256,256), dtype=np.int32)
        semi[c==0]=1
        semi[b>0]=0
        b[b>0]=255
        b[b<0]=255
        inastance_label=morphology.label(b)
        # class labels
        class_id=img[:-4]+'.npy'
        specific_class=np.load(os.path.join(class_path,class_id))[:,:,marker_channel]
        class_label = np.zeros((b.shape[0],b.shape[1]))
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                if specific_class[i,j]>0:
                   class_label[i,j]=2
                else:
                   class_label[i,j]= b[i,j]/255
#         class_label=np.load(os.path.join(class_path,class_id))[:,:,2]+(b/255)    #background=0, nucleu execpt class2 are 1, class 2 is 2
        data=np.concatenate([img_rgb, inastance_label[:,:,np.newaxis], semi[:,:,np.newaxis], class_label[:,:,np.newaxis].astype(np.int32)],2)
        np.save(path_save +img[:-4], data)
        n+=1
        print(n)
        
    

def create_point_label_from_instance(data_dir, save_dir, train_list):
    def get_point(img):
        a = np.where(img != 0)
        rmin, rmax, cmin, cmax = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return (rmin + rmax) // 2, (cmin + cmax) // 2

    create_folder(save_dir)
    print("Generating point label from instance label...")
    image_list = os.listdir(data_dir)
    N_total = len(train_list)
    N_processed = 0
    for image_name in image_list:
        name = image_name.split('.')[0]
        if '{:s}.png'.format(name[:-6]) not in train_list or name[-5:] != 'label':
            continue

        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        image_path = os.path.join(data_dir, image_name)
        image = io.imread(image_path)
        h, w = image.shape

        # extract bbox
        id_max = np.max(image)
        label_point = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, id_max + 1):
            nucleus = image == i
            if np.sum(nucleus) == 0:
                continue
            x, y = get_point(nucleus)
            label_point[x, y] = 255

        io.imsave('{:s}/{:s}_point.png'.format(save_dir, name), label_point.astype(np.uint8))



def generate_colors(class_names):
    hsv_tuples = [(x / class_names, 1., 1.) for x in range(class_names)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def create_Voronoi_label(data_dir, save_dir, train_list, nuclei_label=0):
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon
    from utils import voronoi_finite_polygons_2d, poly2mask

    create_folder(save_dir)
    print("Generating Voronoi label from point label...")
    N_total = len(train_list)
    N_processed = 0
    for img_name in train_list:
        name = img_name.split('.')[0]

        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        img_path = '{:s}/{:s}_label_point.npy'.format(data_dir, name)
        label_point = np.load(img_path)
        h, w = label_point.shape

        points = np.argwhere(label_point > 0)
        vor = Voronoi(points)

        regions, vertices = voronoi_finite_polygons_2d(vor)
        box = Polygon([[0, 0], [0, w], [h, w], [h, 0]])
        region_masks = np.zeros((h, w), dtype=np.int16)
        edges = np.zeros((h, w), dtype=np.bool)
        count = 1
        for region in regions:
            polygon = vertices[region]
            # Clipping polygon
            poly = Polygon(polygon)
            poly = poly.intersection(box)
            polygon = np.array([list(p) for p in poly.exterior.coords])

            mask = poly2mask(polygon[:, 0], polygon[:, 1], (h, w))
            edge = mask * (~morphology.erosion(mask, morphology.disk(1)))
            edges += edge
            region_masks[mask] = count
            count += 1

        # fuse Voronoi edge and dilated points
        label_point_dilated = morphology.dilation(label_point, morphology.disk(2))
        label_vor = np.zeros((h, w, 3), dtype=np.uint8)
        label_vor[:, :, 0] = morphology.closing(edges > 0, morphology.disk(1)).astype(np.uint8) * 255
        label_vor[:, :, 1] = (label_point_dilated > 0).astype(np.uint8) * 255

        io.imsave('{:s}/{:s}_label_vor.png'.format(save_dir, name), label_vor)


def create_cluster_label(data_dir, label_point_dir, label_vor_dir, save_dir_nuclei, train_list, nuclei_channel_image,
                          bit=None, img_dir='None', marker_channel_image=None, Marker_range_dir=None, cluster_label = True, rois_path_=None):

    from scipy.ndimage import morphology as ndi_morph
    create_folder(Marker_range_dir)
    create_folder(save_dir_nuclei)
    print("Generating cluster label from point label...")
    N_total = len(train_list)
    N_processed = 0
    for img_name in train_list:
        name = img_name.split('.')[0]
#         print(name)
        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        # Original images with all z axis, notice it only contains channel nuclei
        ori_image = np.load('{:s}/{:s}.npy'.format(data_dir, name))
        marker_image = io.imread('{:s}/{:s}.tif'.format(img_dir, name))[:, marker_channel_image]
        marker_image = np.moveaxis(marker_image, 0, 2)
        if bit == '16bit':
            ori_image = bytescale(ori_image)
            marker_image = bytescale(marker_image)
        # ROI for marker

        img_size0, img_size1 = marker_image.shape[0], marker_image.shape[1]
        mask_size = np.zeros([img_size0, img_size1])
        roi = read_roi_file(rois_path_ + name + '.roi')
        x = np.array(roi[name]['x'])
        y = np.array(roi[name]['y'])
        n = x.shape[0]
        poly = np.zeros([n, 2], dtype=int)
        poly[:, 0] = x
        poly[:, 1] = y
        mask = Image.new('L', (img_size1, img_size0), 0)  # size = (width, height)
        polygontoadd = poly
        polygontoadd = np.reshape(polygontoadd, np.shape(polygontoadd)[0] * 2)  # convert to [x1,y1,x2,y2,..]
        polygontoadd = polygontoadd.tolist()
        ImageDraw.Draw(mask).polygon(polygontoadd, outline=1, fill=1)
        mask = np.array(mask)
        mask_size += mask
        marker_image = mask_size[:, :, np.newaxis] * marker_image




        # taking the shape of the image
        if len(ori_image.shape) == 3:
            h, w, z = ori_image.shape
        if len(ori_image.shape) == 2:
            h, w, = ori_image.shape
        label_path = '{:s}/{:s}_label_point.npy'.format(label_point_dir, name)  # reading pint labels which contains the corresponding z as well
        label_point = np.load(label_path)
        for i in range(z):
            if i == 0:
                final = label_point[:, :, i]
                final = final[:, :, np.newaxis]
            else:
                a = label_point[:, :, i]
                final = np.concatenate([final, np.zeros((h, w, 10)), a[:, :, np.newaxis]], axis=2)   #TODO: how to choose 10
        # k-means clustering
        dist_embeddings = dist_tranform(255 - final)

        clip_dist_embeddings = np.clip(dist_embeddings, a_min=0, a_max=20)
        for i in range(z):
            if i == 0:
                final_ = clip_dist_embeddings[:, :, i]
                final_ = final_[:, :, np.newaxis]
            else:
                a = clip_dist_embeddings[:, :, i * 10 + 1]      #TODO: how to choose 10
                final_ = np.concatenate([final_, a[:, :, np.newaxis]], axis=2)
        clip_dist_embeddings = final_.reshape(-1, 1)
        # TODO: shape is 2 or 3
        if len(ori_image.shape) == 3:
            color_embeddings = np.array(ori_image, dtype=np.float).reshape(-1, 1) / 10  # Divide by 10 to be at the range of the distance transformation    #
        # TODO: shape is 2 or 3
        if len(ori_image.shape) == 2:
            color_embeddings = np.array(ori_image, dtype=np.float).reshape(-1, 1) / 10
        embeddings = np.concatenate((color_embeddings, clip_dist_embeddings), axis=1)

        # print("\t\tPerforming k-means clustering...")
        # TODO: there is just background and nuclei
        kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)
        clusters = np.reshape(kmeans.labels_, (h, w, z))

        # get nuclei and background clusters
        overlap_nums = [np.sum((clusters == i) * label_point) for i in range(3)]
        nuclei_idx = np.argmax(overlap_nums)
        remain_indices = np.delete(np.arange(3), nuclei_idx)
        # dilated_label_point = morphology.binary_dilation(label_point, morphology.disk(5))
        dilated_label_point = morphology.binary_dilation(label_point, morphology.ball(4))
        overlap_nums = [np.sum((clusters == i) * dilated_label_point) for i in remain_indices]
        background_idx = remain_indices[np.argmin(overlap_nums)]
        nuclei_range = (clusters != background_idx) # it is used to determine the range of nuclei
        # nuclei_cluster = clusters == nuclei_idx
        # background_cluster = clusters == background_idx




        # refine clustering results
        # print("\t\tRefining clustering results...")
        # nuclei_labeled = measure.label(nuclei_cluster)   #TODO:  consider this in voronoi nuclei_cluster (in voronoi)=> cell_i = voronoi_cells == cell_indices[i]  nucleus_i = cell_i[:,:,np.newaxis] * nuclei_cluster;  nucleus_i[:,:,:].sum(axis=0).sum(axis=0)>0
        # initial_nuclei = morphology.remove_small_objects(nuclei_labeled, 10)
        # refined_nuclei = np.zeros(initial_nuclei.shape, dtype=np.bool)
        refined_marker = np.zeros(nuclei_range.shape, dtype=np.float)
        label_vor = io.imread('{:s}/{:s}_label_vor.png'.format(label_vor_dir, img_name[:-4]))
        voronoi_cells = measure.label(label_vor[:, :, 0] == 0)
        voronoi_cells = morphology.dilation(voronoi_cells, morphology.disk(2))



        # refine clustering results
        unique_vals = np.unique(voronoi_cells)
        cell_indices = unique_vals[unique_vals != 0]
        N = len(cell_indices)
        for i in range(N):
            cell_i = voronoi_cells == cell_indices[i]
            # nucleus_i = cell_i[:,:,np.newaxis] * initial_nuclei
            nuclei_range_i = cell_i[:,:,np.newaxis] * nuclei_range
            nuclei_range_cell = nuclei_range_i[:, :, :].sum(axis=0).sum(axis=0) > 0
            marker_image_i = cell_i[:,:,np.newaxis] * marker_image
            marker_image_i = marker_image_i * nuclei_range_cell
            # nucleus_i_dilated = morphology.binary_dilation(nucleus_i, morphology.disk(5))
            # nucleus_i_dilated = morphology.binary_dilation(nucleus_i, morphology.ball(5))
            # nucleus_i_dilated_filled = ndi_morph.binary_fill_holes(nucleus_i_dilated)
            # nucleus_i_final = morphology.binary_erosion(nucleus_i_dilated_filled, morphology.disk(7))
            # nucleus_i_final = morphology.binary_erosion(nucleus_i_dilated_filled, morphology.ball(7))
            # refined_nuclei += nucleus_i_final > 0
            refined_marker += marker_image_i

        # refined_label = np.zeros((h, w, 3), dtype=np.uint8)
        # label_point_dilated = morphology.dilation(label_point, morphology.disk(10))  # Nazanin 10==>15
        # label_point_dilated = morphology.dilation(label_point, morphology.ball(10))
        # refined_label[:, :, 0] = (background_cluster * (refined_nuclei == 0) * (label_point_dilated == 0)).astype(
        #     np.uint8) * 255

        # refined_label[:, :, 1] = refined_nuclei.astype(np.uint8) * 255

        # io.imsave('{:s}/{:s}_label_cluster.png'.format(save_dir_nuclei, name), refined_label)
        np.save('{:s}/{:s}.npy'.format(Marker_range_dir, name), refined_marker)

        if cluster_label==True:
           ori_image_ = ori_image.max(2)
           h, w, = ori_image_.shape
           label_point_ = label_point.max(2)

           # k-means clustering
           dist_embeddings = dist_tranform(255 - label_point_).reshape(-1, 1)
           clip_dist_embeddings = np.clip(dist_embeddings, a_min=0, a_max=20)

           # TODO: shape is 2 or 3
           if len(ori_image_.shape) == 3:
               color_embeddings = np.array(ori_image_, dtype=np.float).reshape(-1, 3) / 10
           # TODO: shape is 2 or 3
           if len(ori_image_.shape) == 2:
               color_embeddings = np.array(ori_image_, dtype=np.float).reshape(-1, 1) / 10
           embeddings = np.concatenate((color_embeddings, clip_dist_embeddings), axis=1)

           # print("\t\tPerforming k-means clustering...")
           # TODO: there is just background and nuclei
           kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)
           clusters = np.reshape(kmeans.labels_, (h, w))

           # get nuclei and background clusters
           overlap_nums = [np.sum((clusters == i) * label_point_) for i in range(3)]
           nuclei_idx = np.argmax(overlap_nums)
           remain_indices = np.delete(np.arange(3), nuclei_idx)
           dilated_label_point = morphology.binary_dilation(label_point_, morphology.disk(5))
           overlap_nums = [np.sum((clusters == i) * dilated_label_point) for i in remain_indices]
           background_idx = remain_indices[np.argmin(overlap_nums)]

           nuclei_cluster = clusters == nuclei_idx
           background_cluster = clusters == background_idx

           # refine clustering results
           # print("\t\tRefining clustering results...")
           nuclei_labeled = measure.label(nuclei_cluster)
           initial_nuclei = morphology.remove_small_objects(nuclei_labeled, 30)
           refined_nuclei = np.zeros(initial_nuclei.shape, dtype=np.bool)

           label_vor = io.imread('{:s}/{:s}_label_vor.png'.format(label_vor_dir, img_name[:-4]))
           voronoi_cells = measure.label(label_vor[:, :, 0] == 0)
           voronoi_cells = morphology.dilation(voronoi_cells, morphology.disk(2))

           # refine clustering results
           unique_vals = np.unique(voronoi_cells)
           cell_indices = unique_vals[unique_vals != 0]
           N = len(cell_indices)

           for i in range(N):
               cell_i = voronoi_cells == cell_indices[i]
               nucleus_i = cell_i * initial_nuclei

               nucleus_i_dilated = morphology.binary_dilation(nucleus_i, morphology.disk(5))
               nucleus_i_dilated_filled = ndi_morph.binary_fill_holes(nucleus_i_dilated)
               nucleus_i_final = morphology.binary_erosion(nucleus_i_dilated_filled, morphology.disk(7))
               refined_nuclei += nucleus_i_final > 0

           refined_label = np.zeros((h, w, 3), dtype=np.uint8)
           label_point_dilated = morphology.dilation(label_point_, morphology.disk(10))
           refined_label[:, :, 0] = (background_cluster * (refined_nuclei == 0) * (label_point_dilated == 0)).astype(
               np.uint8) * 255
           refined_label[:, :, 1] = refined_nuclei.astype(np.uint8) * 255

           io.imsave('{:s}/{:s}_label_cluster.png'.format(save_dir_nuclei, name), refined_label)



def split_patches(data_dir, save_dir, post_fix=None):
    """ split large image into small patches """
    create_folder(save_dir)

    image_list = os.listdir(data_dir)
    for image_name in image_list:
        if image_name.endswith('png'):
            name = image_name.split('.')[0]
            # below condition: if there is a post_fix, it only considers images with the same post_fix
            if post_fix and name[-len(post_fix):] != post_fix:
                continue
            image_path = os.path.join(data_dir, image_name)
            image = io.imread(image_path)
            seg_imgs = []

            # split into 16 patches of size 250x250
            h, w = image.shape[0], image.shape[1]
            patch_size = 256
            h_overlap = 20
            w_overlap = 20
            #             h_overlap = math.ceil((4 * patch_size - h) / 3)
            #             w_overlap = math.ceil((4 * patch_size - w) / 3)
            for i in range(0, h - patch_size + 1, patch_size - h_overlap):
                for j in range(0, w - patch_size + 1, patch_size - w_overlap):
                    if len(image.shape) >= 3:
                        patch = image[i:i + patch_size, j:j + patch_size, :]
                    else:
                        patch = image[i:i + patch_size, j:j + patch_size]
                    #                     print(np.sum(seg_imgs))
                    seg_imgs.append(patch)

            for k in range(len(seg_imgs)):
                if post_fix:
                    io.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(post_fix) - 1], k, post_fix),
                              seg_imgs[k])
                #                     cv2.imwrite('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(post_fix) - 1], k, post_fix), seg_imgs[k])
                else:
                    io.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])
        #                     cv2.imwrite('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])

        if image_name.endswith('npy'):
            name = image_name.split('.')[0]
            # below condition: if there is a post_fix, it only considers images with the same post_fix
            if post_fix and name[-len(post_fix):] != post_fix:
                continue
            image_path = os.path.join(data_dir, image_name)
            image = np.load(image_path)
            seg_imgs = []

            # split into 16 patches of size 250x250
            h, w = image.shape[0], image.shape[1]
            patch_size = 256
            h_overlap = 20
            w_overlap = 20
            #             h_overlap = math.ceil((4 * patch_size - h) / 3)
            #             w_overlap = math.ceil((4 * patch_size - w) / 3)
            for i in range(0, h - patch_size + 1, patch_size - h_overlap):
                for j in range(0, w - patch_size + 1, patch_size - w_overlap):
                    if len(image.shape) >= 3:
                        patch = image[i:i + patch_size, j:j + patch_size, :]
                    else:
                        patch = image[i:i + patch_size, j:j + patch_size]
                    #                     print(np.sum(seg_imgs))
                    seg_imgs.append(patch)

            for k in range(len(seg_imgs)):
                if post_fix:
                    np.save('{:s}/{:s}_{:d}_{:s}.npy'.format(save_dir, name[:-len(post_fix) - 1], k, post_fix),
                            seg_imgs[k])
                #                     cv2.imwrite('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(post_fix) - 1], k, post_fix), seg_imgs[k])
                else:
                    np.save('{:s}/{:s}_{:d}.npy'.format(save_dir, name, k), seg_imgs[k])


#                     cv2.imwrite('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])


def organize_data_for_training(data_dir, train_data_dir,dataset):
    # --- Step 1: create folders --- #
    create_folder(train_data_dir)
    if os.path.exists('{:s}/images'.format(train_data_dir)):
        shutil.rmtree('{:s}/images'.format(train_data_dir))
    if os.path.exists('{:s}/labels_point'.format(train_data_dir)):
        shutil.rmtree('{:s}/labels_point'.format(train_data_dir))
    if os.path.exists('{:s}/images_RGB'.format(train_data_dir)):
        shutil.rmtree('{:s}/images_RGB'.format(train_data_dir))
    if os.path.exists('{:s}/labels_voronoi'.format(train_data_dir)):
        shutil.rmtree('{:s}/labels_voronoi'.format(train_data_dir))
    if os.path.exists('{:s}/labels_cluster'.format(train_data_dir)):
        shutil.rmtree('{:s}/labels_cluster'.format(train_data_dir))
    if os.path.exists('{:s}/classification_labels'.format(train_data_dir)):
        shutil.rmtree('{:s}/classification_labels'.format(train_data_dir))

    create_folder('{:s}/images/train'.format(train_data_dir))
    create_folder('{:s}/images/val'.format(train_data_dir))
    create_folder('{:s}/images/test'.format(train_data_dir))

    create_folder('{:s}/classification_labels/train'.format(train_data_dir))
    create_folder('{:s}/classification_labels/val'.format(train_data_dir))
    create_folder('{:s}/classification_labels/test'.format(train_data_dir))

    create_folder('{:s}/images_RGB/train'.format(train_data_dir))
    create_folder('{:s}/images_RGB/val'.format(train_data_dir))
    create_folder('{:s}/images_RGB/test'.format(train_data_dir))

    create_folder('{:s}/labels_point/train'.format(train_data_dir))
    create_folder('{:s}/labels_point/val'.format(train_data_dir))
    create_folder('{:s}/labels_point/test'.format(train_data_dir))

    create_folder('{:s}/labels_voronoi/train'.format(train_data_dir))
    create_folder('{:s}/labels_cluster/train'.format(train_data_dir))
    create_folder('{:s}/labels_voronoi/val'.format(train_data_dir))
    create_folder('{:s}/labels_cluster/val'.format(train_data_dir))
    create_folder('{:s}/labels_voronoi/test'.format(train_data_dir))
    create_folder('{:s}/labels_cluster/test'.format(train_data_dir))

    # --- Step 2: move images and labels to each folder --- #
    print('Organizing data for training...')
    with open('../Image_classification/{:s}/train_val_test.json'.format(dataset), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    # train
    for img_name in train_list:
        name = img_name.split('.')[0]
        # images
        for file in glob.glob('{:s}/patches/images/{:s}*'.format(data_dir, name)):
            if np.sum(io.imread(file)) > 0:
                file_name = file.split('/')[-1]
                dst = '{:s}/images/train/{:s}'.format(train_data_dir, file_name)
                shutil.copyfile(file, dst)
                # label_voronoi
                file_vor = '{:s}/patches/labels_voronoi/{:s}'.format(data_dir, file_name[:-4] + '_label_vor.png')
                dst = '{:s}/labels_voronoi/train/{:s}'.format(train_data_dir, file_vor.split('/')[-1])
                shutil.copyfile(file_vor, dst)
                # label_cluster
                file_clus = '{:s}/patches/labels_cluster/{:s}'.format(data_dir, file_name[:-4] + '_label_cluster.png')
                dst = '{:s}/labels_cluster/train/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
                shutil.copyfile(file_clus, dst)
                # label_point
                file_clus = '{:s}/patches/labels_point/{:s}'.format(data_dir, file_name[:-4] + '_label_point.npy')
                dst = '{:s}/labels_point/train/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
                shutil.copyfile(file_clus, dst)

#                 file_clus = '{:s}/patches/images_RGB/{:s}'.format(data_dir, file_name[:-4] + '.png')
#                 dst = '{:s}/images_RGB/train/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
#                 shutil.copyfile(file_clus, dst)

                file_clus = '{:s}/patches/classification_labels/{:s}'.format(data_dir, file_name[:-4] + '.npy')
                dst = '{:s}/classification_labels/train/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
                shutil.copyfile(file_clus, dst)

    # val
    for img_name in val_list:
        name = img_name.split('.')[0]
        # images
        for file in glob.glob('{:s}/patches/images/{:s}*'.format(data_dir, name)):
            if np.sum(io.imread(file)) > 0:
                file_name = file.split('/')[-1]
                dst = '{:s}/images/val/{:s}'.format(train_data_dir, file_name)
                shutil.copyfile(file, dst)
                # label_voronoi
                file_vor = '{:s}/patches/labels_voronoi/{:s}'.format(data_dir,
                                                                     file_name[:-4] + '_label_vor.png')
                dst = '{:s}/labels_voronoi/val/{:s}'.format(train_data_dir, file_vor.split('/')[-1])
                shutil.copyfile(file_vor, dst)
                # label_cluster
                file_clus = '{:s}/patches/labels_cluster/{:s}'.format(data_dir,
                                                                      file_name[:-4] + '_label_cluster.png')
                dst = '{:s}/labels_cluster/val/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
                shutil.copyfile(file_clus, dst)

                # label_point
                file_clus = '{:s}/patches/labels_point/{:s}'.format(data_dir, file_name[:-4] + '_label_point.npy')
                dst = '{:s}/labels_point/val/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
                shutil.copyfile(file_clus, dst)

#                 file_clus = '{:s}/patches/images_RGB/{:s}'.format(data_dir, file_name[:-4] + '.png')
#                 dst = '{:s}/images_RGB/val/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
#                 shutil.copyfile(file_clus, dst)

                file_clus = '{:s}/patches/classification_labels/{:s}'.format(data_dir, file_name[:-4] + '.npy')
                dst = '{:s}/classification_labels/val/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
                shutil.copyfile(file_clus, dst)

    # test
    for img_name in test_list:
        name = img_name.split('.')[0]
        # images
        for file in glob.glob('{:s}/patches/images/{:s}*'.format(data_dir, name)):
            if np.sum(io.imread(file)) > 0:
                file_name = file.split('/')[-1]
                dst = '{:s}/images/test/{:s}'.format(train_data_dir, file_name)
                shutil.copyfile(file, dst)
                # label_voronoi
                file_vor = '{:s}/patches/labels_voronoi/{:s}'.format(data_dir,
                                                                     file_name[:-4] + '_label_vor.png')
                dst = '{:s}/labels_voronoi/test/{:s}'.format(train_data_dir, file_vor.split('/')[-1])
                shutil.copyfile(file_vor, dst)
                # label_cluster
                file_clus = '{:s}/patches/labels_cluster/{:s}'.format(data_dir,
                                                                      file_name[:-4] + '_label_cluster.png')
                dst = '{:s}/labels_cluster/test/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
                shutil.copyfile(file_clus, dst)

                # label_point
                file_clus = '{:s}/patches/labels_point/{:s}'.format(data_dir, file_name[:-4] + '_label_point.npy')
                dst = '{:s}/labels_point/test/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
                shutil.copyfile(file_clus, dst)

#                 file_clus = '{:s}/patches/images_RGB/{:s}'.format(data_dir, file_name[:-4] + '.png')
#                 dst = '{:s}/images_RGB/test/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
#                 shutil.copyfile(file_clus, dst)

                file_clus = '{:s}/patches/classification_labels/{:s}'.format(data_dir, file_name[:-4] + '.npy')
                dst = '{:s}/classification_labels/test/{:s}'.format(train_data_dir, file_clus.split('/')[-1])
                shutil.copyfile(file_clus, dst)

        # # label_voronoi
        # for file in glob.glob('{:s}/patches/labels_voronoi/{:s}*'.format(data_dir, name)):
        #     file_name = file.split('/')[-1]
        #     dst = '{:s}/labels_voronoi/train/{:s}'.format(train_data_dir, file_name)
        #     shutil.copyfile(file, dst)
        # label_cluster
        # for file in glob.glob('{:s}/patches/labels_cluster/{:s}*'.format(data_dir, name)):
        #     file_name = file.split('/')[-1]
        #     dst = '{:s}/labels_cluster/train/{:s}'.format(train_data_dir, file_name)
        #     shutil.copyfile(file, dst)
    # val


#     for img_name in val_list:
#         name = img_name.split('.')[0]
#         # images
#         for file in glob.glob('{:s}/images/{:s}*'.format(data_dir, name)):
#             file_name = file.split('/')[-1]
#             dst = '{:s}/images/val/{:s}'.format(train_data_dir, file_name)
#             shutil.copyfile(file, dst)
#     # test
#     for img_name in test_list:
#         name = img_name.split('.')[0]
#         # images
#         for file in glob.glob('{:s}/images/{:s}*'.format(data_dir, name)):
#             file_name = file.split('/')[-1]
#             dst = '{:s}/images/test/{:s}'.format(train_data_dir, file_name)
#             shutil.copyfile(file, dst)


def compute_mean_std(data_dir, train_data_dir, dataset):
    """ compute mean and standarad deviation of training images """
    total_sum = np.zeros(3)  # total sum of all pixel values in each channel
    total_square_sum = np.zeros(3)
    num_pixel = 0  # total num of all pixels

    with open('../Image_classification/{:s}/train_val_test.json'.format(dataset), 'r') as file:
        data_list = json.load(file)
        train_list = data_list['train']

    print('Computing the mean and standard deviation of training data...')

    for file_name in train_list:
        img_name = '{:s}/{:s}'.format(data_dir, file_name)
        img = io.imread(img_name)
        if len(img.shape) != 3 or img.shape[2] < 3:
            continue
        img = img[:, :, :3].astype(int)
        total_sum += img.sum(axis=(0, 1))
        total_square_sum += (img ** 2).sum(axis=(0, 1))
        num_pixel += img.shape[0] * img.shape[1]

    # compute the mean values of each channel
    mean_values = total_sum / num_pixel

    # compute the standard deviation
    std_values = np.sqrt(total_square_sum / num_pixel - mean_values ** 2)

    # normalization
    mean_values = mean_values / 255
    std_values = std_values / 255

    np.save('{:s}/mean_std.npy'.format(train_data_dir), np.array([mean_values, std_values]))
    np.savetxt('{:s}/mean_std.txt'.format(train_data_dir), np.array([mean_values, std_values]), '%.4f', '\t')
    # np.save('{:s}/mean_std.npy'.format(train_data_dir), np.array([mean_values, std_values]))


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def instance_labels(label_cluster_dir, dataset, train_list,n_channel, gt_class_label):
    data_annotate = pd.read_excel('../Image_classification/{:s}/SpreadsheetforCellSummary.xlsx'.format(dataset))
    data_annotate.columns = data_annotate.iloc[0]
    data_annotate = data_annotate.drop(0)
    create_folder(gt_class_label)
    print("Generating cluster label from point label...")
    N_total = len(train_list)
    N_processed = 0
    list_img = []
    for img_name in train_list:
        name = img_name.split('.')[0]
        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)
        annotation = data_annotate[data_annotate.Filename.str.startswith(name)]
        # print('\t[{:d}/{:d}] Processing image {:s} ...'.format(count, len(img_list), img_name))
        image_rgb = io.imread('{:s}/{:s}_label_cluster.png'.format(label_cluster_dir, name))
        RGB_image = morphology.label(np.array(image_rgb)[:, :, 1])  # for green masks
        #     print(name_image, RGB_image.shape)
        img_size0, img_size1 = RGB_image.shape[0], RGB_image.shape[1]
        label = np.zeros((img_size0, img_size1, n_channel))
        for j in range(annotation.shape[0]):
            mask = np.zeros([img_size0, img_size1])
            x_ = int(annotation.iloc[j].X[0])
            y_ = -int(annotation.iloc[j].Y[0])  # Y is negative
            mask[y_, x_] = 1
            instance_number = (mask * RGB_image).max()
            instance_selected = RGB_image == instance_number
            if instance_number > 0:  # some of the points are not in the data
                n_labels = len(str(annotation.iloc[j].Category))  # number of labels
                #                 print(n_labels, instance_number, x_, y_)
                for n in range(n_labels):
                    #                   print(str(annotation.iloc[j].Category)[n])
                    class_number = str(annotation.iloc[j].Category)[n]
                    if n_labels == 1:
                        #                      print('1', n_labels)
                        #                      print(instance_selected.max())
                        #                      print('label1',label.max())
                        label[:, :,
                        int(class_number) - 1] += instance_selected  # if the nuclei does not have any markers arround
                    #                      print('label1',label.max())
                    else:

                        if int(class_number) != 1:  # 1 is corresponding to nuclei
                            label[:, :,
                            int(class_number) - 1] += instance_selected  # types of the nuclei based on markers arround them

            #                 print(instance_number)
            #                 plt.figure(figsize=(5,5))
            #                 plt.axis('off')
            #                 plt.imshow(instance_selected)
            #                 plt.show()
        label[label > 1] = 1  # a couple of max are corresponding t more than one nuclei
        np.save(os.path.join(gt_class_label , name + '.npy'), label)

def nuclei_point(img_dir, label_point_dir_2D, label_point_dir_3D, dataset, train_list, nuclei_channel_image, rois_path, ROI_nuclei_dir):
    """
    This fuction save the 2D and 3D masks for images
    as well as nuclei channel (Region of interest)

    Output: 2D and 3D masks and nuclei channel (ROI)
    """

    create_folder(label_point_dir_2D)
    create_folder(label_point_dir_3D)
    create_folder(ROI_nuclei_dir)
    data_annotate = pd.read_excel('../Image_classification/{:s}/SpreadsheetforCellSummary.xlsx'.format(dataset))
    data_annotate.columns = data_annotate.iloc[0]
    data_annotate = data_annotate.drop(0)

    print("Generating cluster label from point label...")
    N_total = len(train_list)
    N_processed = 0
    list_img = []
    for img_name in train_list:
        name = img_name.split('.')[0]
        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        # print('\t[{:d}/{:d}] Processing image {:s} ...'.format(count, len(img_list), img_name))
        ori_image = io.imread('{:s}/{:s}.tif'.format(img_dir, name))[:, nuclei_channel_image]
        ori_image = np.moveaxis(ori_image, 0, 2)  # move z to the end
        ROI(img_dir, img_name, nuclei_channel_image, rois_path, ROI_nuclei_dir)
        if len(ori_image.shape) == 3:
            h, w, z = ori_image.shape
        if len(ori_image.shape) == 2:
            h, w = ori_image.shape
        annotation = data_annotate[data_annotate.Filename.str.startswith(name)]
        mask_3D = np.zeros([h, w, z])
        mask_2D = np.zeros([h, w])
        for j in range(annotation.shape[0]):
            x_ = int(annotation.iloc[j].X[0])
            y_ = -int(annotation.iloc[j].Y[0])  # Y is negative
            z_ = annotation.iloc[j]['Z-slice position']
            mask_3D[y_, x_, z_-1] = 255    # it is ranged in the exel file from 1 to 11
            mask_2D[y_, x_] = 255
            list_img.append(x_)  # just for checking the number of annotations

        np.save(label_point_dir_2D + name + '_label_point.npy', mask_2D)
        np.save(label_point_dir_3D + name + '_label_point.npy', mask_3D)

        g_mask = gaussian_filter(mask_2D, sigma=5)
        # plt.figure(figsize=(5, 5))
        # plt.axis('off')
        # plt.imshow(g_mask)
        # plt.show()
        #
        # plt.figure(figsize=(5, 5))
        # plt.axis('off')
        # plt.imshow(np.max(ori_image,axis=2))
        # plt.show()

def ROI(img_dir, img_name, nuclei_channel_image, rois_path, ROI_nuclei_dir):
    name = img_name.split('.')[0]
    ori_image = io.imread('{:s}/{:s}.tif'.format(img_dir, name))[:, nuclei_channel_image]
    ori_image = np.moveaxis(ori_image, 0, 2)  # move z to the end
    img_size0, img_size1 = ori_image.shape[0], ori_image.shape[1]
    mask_size = np.zeros([img_size0, img_size1])
    roi = read_roi_file(rois_path + name + '.roi')
    x = np.array(roi[name]['x'])
    y = np.array(roi[name]['y'])
    n = x.shape[0]
    poly = np.zeros([n, 2], dtype=int)
    poly[:, 0] = x
    poly[:, 1] = y
    mask = Image.new('L', (img_size1, img_size0), 0)  # size = (width, height)
    polygontoadd = poly
    polygontoadd = np.reshape(polygontoadd, np.shape(polygontoadd)[0] * 2)  # convert to [x1,y1,x2,y2,..]
    polygontoadd = polygontoadd.tolist()
    ImageDraw.Draw(mask).polygon(polygontoadd, outline=1, fill=1)
    mask = np.array(mask)
    mask_size += mask
    reagion_interest = mask_size[:, :, np.newaxis] * ori_image
    np.save(ROI_nuclei_dir + name + '.npy', reagion_interest)

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, default=None
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, default=None
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, default=None
        Scale max value to `high`.  Default is 255.
    low : scalar, default=None
        Scale min value to `low`.  Default is 0.
    Returns

    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

if __name__ == '__main__':
    opt = Options(isTrain=True)
    opt.parse()
    main(opt)
