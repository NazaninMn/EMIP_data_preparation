
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import skimage.morphology as morph
import scipy.ndimage.morphology as ndi_morph
from skimage import measure, io

from model import create_model
import utils
from accuracy import compute_metrics
import json
from options import Options
from my_transforms import get_transforms
from accuracy import compute_accuracy
import cv2


r1 = 15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
main_root = '/scratch/nm4wu/AHA/'




def main(opt, group):
    total_TP = 0.0
    total_FP = 0.0
    total_FN = 0.0
    total_d_list = []
    test_results = dict()
    dataset = opt.dataset
    data_dir = '../data/{:s}'.format(dataset)
    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    opt.isTrain = False

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])
    if group=='test_wsi':
        img_dir = opt.test['img_dir']
        label_dir = '../data/MO/labels_point'
        img_names=test_list
    if group == 'train_wsi':
        img_dir = '../data/MO/images'
        label_dir = '../data/MO/labels_point'
        img_names=train_list
    if group == 'new_data':
        img_dir = '../data_for_train/MO/images/new_data'
        label_dir = '../data_for_train/MO/labels_detect/new_data'
        img_names=os.listdir('../data_for_train/MO/images/new_data')

    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']
    save_flag = opt.test['save_flag']
    if save_flag and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    opt.save_options()

    # data transforms
    test_transform = get_transforms(opt.transform['test'])

    model = create_model(opt.model['name'], opt.model['out_c'], opt.model['pretrained'])
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print("=> loading trained model")
    # TODO: for now I upload checkpoint
    # checkpoint = torch.load(model_path)
    model.load_state_dict(torch.load('/scratch/nm4wu/AHA/WeaklySegPartialPoints/code_seg/model_trained_segmentation'))
    # print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module

    # switch to evaluate mode
    model.eval()
    counter = 0
    print("=> Test begins:")

    # img_names = os.listdir(img_dir)

    if save_flag:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        strs = img_dir.split('/')
        prob_maps_folder = '{:s}/{:s}_prob_maps'.format(save_dir, strs[-1])
        seg_folder = '{:s}/{:s}_segmentation'.format(save_dir, strs[-1])
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        if not os.path.exists(seg_folder):
            os.mkdir(seg_folder)

    metric_names = ['acc', 'p_F1', 'dice', 'aji']
    all_result = utils.AverageMeter(len(metric_names))

    for img_name in img_names:
        # load test image
        # print('=> Processing image {:s}'.format(img_name))
        if img_name.endswith('png'):
            img_path = '{:s}/{:s}'.format(img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            ori_h = img.size[1]
            ori_w = img.size[0]

            name = os.path.splitext(img_name)[0]
            if group=='new_data':
                label_path = '{:s}/{:s}_Stitch_label_point.png'.format(os.path.join(main_root, 'WeaklySegPartialPoints/data_for_train/MO/labels_detect/new_data_point_annotations'),name)
            else:
                label_path = '{:s}/{:s}_label_point.png'.format(label_dir, name)
            gt = io.imread(label_path)

            input = test_transform((img,))[0].unsqueeze(0)

            # print('\tComputing output probability maps...')
            prob_maps = utils.get_probmaps(input, model, opt)
            pred = np.argmax(prob_maps, axis=0)  # prediction

            pred_labeled = measure.label(pred)
            pred_labeled = morph.remove_small_objects(pred_labeled, opt.post['min_area'])
            pred_labeled = ndi_morph.binary_fill_holes(pred_labeled > 0)
            pred_labeled, N = measure.label(pred_labeled, return_num=True)

            # print('\tComputing metrics...')
            TP, FP, FN, d_list, pred_points, true_pos = compute_accuracy(pred_labeled, gt, radius=r1, return_distance=True, test=True)
            if not os.path.exists(os.path.join(main_root, 'results/' + group + '/')):
                os.mkdir(os.path.join(main_root, 'results/' + group + '/'))
            cv2.imwrite(os.path.join(main_root, 'results/' + group + '/') + name + '_all_points.png',
                        ((pred_labeled > 0) * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(main_root, 'results/' + group + '/') + name + '_prob_map.png',
                        pred.astype(np.uint8))
            mask = np.zeros((pred.shape[0], pred.shape[1]))
            for i in range(pred_points.shape[0]):
                mask[int(pred_points[i, 0]), int(pred_points[i, 1])] = 1
            cv2.imwrite(os.path.join(main_root, 'results/' + group + '/') + name + '_False_positive.png',
                        ((mask > 0) * 255).astype(np.uint8))
            mask = np.zeros((pred.shape[0], pred.shape[1]))
            true_pos = np.asarray(true_pos)
            for i in range(true_pos.shape[0]):
                mask[int(true_pos[i, 0]), int(true_pos[i, 1])] = 1
            cv2.imwrite(os.path.join(main_root, 'results/' + group + '/') + name + '_True_positive.png',
                        ((mask > 0) * 255).astype(np.uint8))
            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_d_list += d_list
            # save result for each image
            test_results[name] = [float(TP) / (TP + FP + 1e-8), float(TP) / (TP + FN + 1e-8),
                                  float(2 * TP) / (2 * TP + FP + FN + 1e-8)]

    recall = float(total_TP) / (total_TP + total_FN + 1e-8)
    precision = float(total_TP) / (total_TP + total_FP + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    if len(total_d_list) > 0:
        mu = np.mean(np.array(total_d_list))
        sigma = np.sqrt(np.var(np.array(total_d_list)))
    else:
        mu = -1
        sigma = -1

    print('Average: precision\trecall\tF1\tmean\tstd:'
          '\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:3f}\t{:.3f}'.format(precision, recall, F1, mu, sigma))

    header = ['precision', 'recall', 'F1', 'mean', 'std']
    strs = img_dir.split('/')
    save_results(header, [precision, recall, F1, mu, sigma], test_results,'{}/results/{}/{}_test_result_.txt'.format(main_root, group, strs[-1]))

    return recall, precision, F1

        # metrics = compute_metrics(pred_labeled, gt, metric_names)

    #         # save result for each image
    #         test_results[name] = [metrics['acc'], metrics['p_F1'], metrics['dice'], metrics['aji']]
    #
    #         # update the average result
    #         all_result.update([metrics['acc'], metrics['p_F1'], metrics['dice'], metrics['aji']])
    #
    #         # save image
    #         if save_flag:
    #             # print('\tSaving image results...')
    #             io.imsave('{:s}/{:s}_pred.png'.format(prob_maps_folder, name), (pred_labeled>0).astype(np.uint8) * 255)
    #             io.imsave('{:s}/{:s}_prob.png'.format(prob_maps_folder, name), prob_maps[1, :, :])
    #             final_pred = Image.fromarray(pred_labeled.astype(np.uint16))
    #             final_pred.save('{:s}/{:s}_seg.tiff'.format(seg_folder, name))
    #
    #             # save colored objects
    #             pred_colored_instance = np.zeros((ori_h, ori_w, 3))
    #             for k in range(1, pred_labeled.max() + 1):
    #                 pred_colored_instance[pred_labeled == k, :] = np.array(utils.get_random_color())
    #             filename = '{:s}/{:s}_seg_colored.png'.format(seg_folder, name)
    #             io.imsave(filename, pred_colored_instance)
    #
    #         counter += 1
    #         if counter % 10 == 0:
    #             print('\tProcessed {:d} images'.format(counter))
    #
    # print('=> Processed all {:d} images'.format(counter))
    # print('Average Acc: {r[0]:.4f}\nF1: {r[1]:.4f}\nDice: {r[2]:.4f}\nAJI: {r[3]:.4f}\n'.format(r=all_result.avg))
    #
    # header = metric_names
    # utils.save_results(header, all_result.avg, test_results, '{:s}/test_results.txt'.format(save_dir))

def save_results(header, avg_result, test_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    assert N == len(avg_result)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(avg_result[i]))
        file.write('{:.4f}\n'.format(avg_result[N - 1]))

        # results for each image
        for key, vals in sorted(test_results.items()):
            file.write('{:s}:\t'.format(key))
            for i in range(len(vals)-1):
                file.write('{:.4f}\t'.format(vals[i]))
            file.write('{:.4f}\n'.format(vals[-1]))


if __name__ == '__main__':
    opt = Options(isTrain=False)
    opt.parse()
    opt.print_options()
    main(opt, group='test_wsi')
