import os
import numpy as np
import argparse


class Options:
    def __init__(self, isTrain):
        self.isTrain = isTrain  # train or test mode
        self.dataset = 'OL_LGALS3'     # dataset: LC: Lung Cancer, MO: MultiOrgan
        self.ratio = 1
        self.radius = 15
        self.nuclei_channel = 0
        self.nuclei_channel_image = 2
        self.bit = '16bit'
        self.marker_channel_image = 3 #lineage tracing:1, LGALS3:3
        self.marker_channel = 3 #lineage tracing:2, LGALS3:3
        self.detection_results_dir = None       # prepare labels from detected points if not None


#         # --- training params --- #
        self.train = dict()
        self.train['random_seed'] = 1234
        self.train['data_dir'] = '../data_for_train/{:s}'.format(self.dataset)  # path to data
        self.train['save_dir'] = '../experiments/segmentation/{:s}/{:.2f}/1'.format(self.dataset, self.ratio)  
        self.train['gpus'] = [0, ]              # select gpu devices



    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--detection-results-dir', type=str, default=self.detection_results_dir, help='detected points')
            parser.add_argument('--gpus', type=int, nargs='+', default=self.train['gpus'], help='GPUs for training')
            args = parser.parse_args()

            self.detection_results_dir = args.detection_results_dir
            self.train['gpus'] = args.gpus
