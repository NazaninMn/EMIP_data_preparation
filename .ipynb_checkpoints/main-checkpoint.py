import os, json
import torch
import numpy as np
import random

from options import Options
from prepare_data import main as prepare_data
from train import main as train
from test import main as test


def main():
    opt = Options(isTrain=True)
    opt.parse()
    # opt.print_options()

    if opt.train['random_seed'] >= 0:
        print('=> Using random seed {:d}'.format(opt.train['random_seed']))
        torch.manual_seed(opt.train['random_seed'])
        torch.cuda.manual_seed(opt.train['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.train['random_seed'])
        random.seed(opt.train['random_seed'])
    else:
        torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    print('=============== training: ratio={:.2f} ==============='.format(opt.ratio))
    # opt.train['save_dir'] = '../experiments/segmentation/{:s}/{:.2f}'.format(opt.dataset,  opt.ratio)
    os.makedirs(opt.train['save_dir'], exist_ok=True)

    # ----- prepare training data ----- #
    print('=> Preparing training samples')
    # detection_results_dir = '../experiments/detection/{:s}/{:.2f}/3/best/images_prob_maps'.format(opt.dataset, opt.ratio)
    prepare_data(opt)



if __name__ == '__main__':
    main()
