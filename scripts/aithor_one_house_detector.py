from ai2thor.controller import Controller
import ipdb
st = ipdb.set_trace
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
import time
import quaternion
import math
import pickle
from scipy.spatial.transform import Rotation as R
# from ai2thor_docker.x_server import startx

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model

# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import Image
from tensorboardX import SummaryWriter

import torchvision
from torchvision import transforms
import PIL
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

import subprocess
import time
import shlex
import re
import atexit
import platform
import tempfile
import threading
import os
import sys

# import psutil
# import signal 

import torch
import torch.nn as nn
import torch.nn.functional as F 
from argparse import Namespace

# from nets.cemnet import CEMNet
from nets.localpnet import LocalPNET
import utils.improc
from utils.utils import Utils

class Ai2Thor():
    def __init__(self):   
        self.visualize = False
        self.verbose = False
        self.save_imgs = True

        self.plot_loss = True
        # st()

        mapnames = []
        for i in [1, 201, 301, 401]:
            mapname = 'FloorPlan' + str(i)
            mapnames.append(mapname)

        # random.shuffle(mapnames)
        self.mapnames_train = mapnames
        self.num_episodes = len(self.mapnames_train )   

        # get rest of the house in orders
        a = np.arange(2, 30)
        b = np.arange(202, 231)
        c = np.arange(302, 331)
        d = np.arange(402, 431)
        abcd = np.hstack((a,b,c,d))
        mapnames = []
        for i in range(a.shape[0]):
            mapname = 'FloorPlan' + str(a[i])
            mapnames.append(mapname)
            mapname = 'FloorPlan' + str(b[i])
            mapnames.append(mapname)
            mapname = 'FloorPlan' + str(c[i])
            mapnames.append(mapname)
            mapname = 'FloorPlan' + str(d[i])
            mapnames.append(mapname)

        self.mapnames_test = mapnames

        self.ignore_classes = []  
        # classes to save   
        self.include_classes = [
            'ShowerDoor', 'Cabinet', 'CounterTop', 'Sink', 'Towel', 'HandTowel', 'TowelHolder', 'SoapBar', 
            'ToiletPaper', 'ToiletPaperHanger', 'HandTowelHolder', 'SoapBottle', 'GarbageCan', 'Candle', 'ScrubBrush', 
            'Plunger', 'SinkBasin', 'Cloth', 'SprayBottle', 'Toilet', 'Faucet', 'ShowerHead', 'Box', 'Bed', 'Book', 
            'DeskLamp', 'BasketBall', 'Pen', 'Pillow', 'Pencil', 'CellPhone', 'KeyChain', 'Painting', 'CreditCard', 
            'AlarmClock', 'CD', 'Laptop', 'Drawer', 'SideTable', 'Chair', 'Blinds', 'Desk', 'Curtains', 'Dresser', 
            'Watch', 'Television', 'WateringCan', 'Newspaper', 'FloorLamp', 'RemoteControl', 'HousePlant', 'Statue', 
            'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'BaseballBat', 'TennisRacket', 'VacuumCleaner', 'Mug', 'ShelvingUnit', 
            'Shelf', 'StoveBurner', 'Apple', 'Lettuce', 'Bottle', 'Egg', 'Microwave', 'CoffeeMachine', 'Fork', 'Fridge', 
            'WineBottle', 'Spatula', 'Bread', 'Tomato', 'Pan', 'Cup', 'Pot', 'SaltShaker', 'Potato', 'PepperShaker', 
            'ButterKnife', 'StoveKnob', 'Toaster', 'DishSponge', 'Spoon', 'Plate', 'Knife', 'DiningTable', 'Bowl', 
            'LaundryHamper', 'Vase', 'Stool', 'CoffeeTable', 'Poster', 'Bathtub', 'TissueBox', 'Footstool', 'BathtubBasin', 
            'ShowerCurtain', 'TVStand', 'Boots', 'RoomDecor', 'PaperTowelRoll', 'Ladle', 'Kettle', 'Safe', 'GarbageBag', 'TeddyBear', 
            'TableTopDecor', 'Dumbbell', 'Desktop', 'AluminumFoil', 'Window']

        self.maskrcnn_to_ithor  = {
            1:'ShowerDoor', 2:'Cabinet', 3:'CounterTop', 4:'Sink', 5:'Towel', 6:'HandTowel', 7:'TowelHolder', 8:'SoapBar', 
            9:'ToiletPaper', 10:'ToiletPaperHanger', 11:'HandTowelHolder', 12:'SoapBottle', 13:'GarbageCan', 14:'Candle', 15:'ScrubBrush', 
            16:'Plunger', 17:'SinkBasin', 18:'Cloth', 19:'SprayBottle', 20:'Toilet', 21:'Faucet', 22:'ShowerHead', 23:'Box', 24:'Bed', 25:'Book', 
            26:'DeskLamp', 27:'BasketBall', 28:'Pen', 29:'Pillow', 30:'Pencil', 31:'CellPhone', 32:'KeyChain', 33:'Painting', 34:'CreditCard', 
            35:'AlarmClock', 36:'CD', 37:'Laptop', 38:'Drawer', 39:'SideTable', 40:'Chair', 41:'Blinds', 42:'Desk', 43:'Curtains', 44:'Dresser', 
            45:'Watch', 46:'Television', 47:'WateringCan', 48:'Newspaper', 49:'FloorLamp', 50:'RemoteControl', 51:'HousePlant', 52:'Statue', 
            53:'Ottoman', 54:'ArmChair', 55:'Sofa', 56:'DogBed', 57:'BaseballBat', 58:'TennisRacket', 59:'VacuumCleaner', 60:'Mug', 61:'ShelvingUnit', 
            62:'Shelf', 63:'StoveBurner', 64:'Apple', 65:'Lettuce', 66:'Bottle', 67:'Egg', 68:'Microwave', 69:'CoffeeMachine', 70:'Fork', 71:'Fridge', 
            72:'WineBottle', 73:'Spatula', 74:'Bread', 75:'Tomato', 76:'Pan', 77:'Cup', 78:'Pot', 79:'SaltShaker', 80:'Potato', 81:'PepperShaker', 
            82:'ButterKnife', 83:'StoveKnob', 84:'Toaster', 85:'DishSponge', 86:'Spoon', 87:'Plate', 88:'Knife', 89:'DiningTable', 90:'Bowl', 
            91:'LaundryHamper', 92:'Vase', 93:'Stool', 94:'CoffeeTable', 95:'Poster', 96:'Bathtub', 97:'TissueBox', 98:'Footstool', 99:'BathtubBasin', 
            100:'ShowerCurtain', 101:'TVStand', 102:'Boots', 103:'RoomDecor', 104:'PaperTowelRoll', 105:'Ladle', 106:'Kettle', 107:'Safe', 108:'GarbageBag', 109:'TeddyBear', 
            110:'TableTopDecor', 111:'Dumbbell', 112:'Desktop', 113:'AluminumFoil', 114:'Window'}

        self.ithor_to_maskrcnn = {value:key for key, value in self.maskrcnn_to_ithor.items()}

        # # These are all classes shared between aithor and coco
        # self.include_classes = [
        #     'Sink', 
        #     'Toilet', 'Bed', 'Book', 
        #     'CellPhone', 
        #     'AlarmClock', 'Laptop', 'Chair',
        #     'Television', 'RemoteControl', 'HousePlant', 
        #     'Ottoman', 'ArmChair', 'Sofa', 'BaseballBat', 'TennisRacket', 'Mug', 
        #     'Apple', 'Bottle', 'Microwave', 'Fork', 'Fridge', 
        #     'WineBottle', 'Cup', 
        #     'ButterKnife', 'Toaster', 'Spoon', 'Knife', 'DiningTable', 'Bowl', 
        #     'Vase', 
        #     'TeddyBear', 
        #     ]


        # self.maskrcnn_to_ithor = {
        #     81:'Sink', 
        #     70:'Toilet', 65:'Bed', 84:'Book', 
        #     77:'CellPhone', 
        #     85:'AlarmClock', 73:'Laptop', 62:'Chair',
        #     72:'Television', 75:'RemoteControl', 64:'HousePlant', 
        #     62:'Ottoman', 62:'ArmChair', 63:'Sofa', 39:'BaseballBat', 43:'TennisRacket', 47:'Mug', 
        #     53:'Apple', 44:'Bottle', 78:'Microwave', 48:'Fork', 82:'Fridge', 
        #     44:'WineBottle', 47:'Cup', 
        #     49:'ButterKnife', 80:'Toaster', 50:'Spoon', 49:'Knife', 67:'DiningTable', 51:'Bowl', 
        #     86:'Vase', 
        #     88:'TeddyBear', 
        # }

        # self.ithor_to_maskrcnn = {
        #     'Sink':81, 
        #     'Toilet':70, 'Bed':65, 'Book':84, 
        #     'CellPhone':77, 
        #     'AlarmClock':85, 'Laptop':73, 'Chair':62,
        #     'Television':72, 'RemoteControl':75, 'HousePlant':64, 
        #     'Ottoman':62, 'ArmChair':62, 'Sofa':63, 'BaseballBat':39, 'TennisRacket':43, 'Mug':47, 
        #     'Apple':53, 'Bottle':44, 'Microwave':78, 'Fork':48, 'Fridge':82, 
        #     'WineBottle':44, 'Cup':47, 
        #     'ButterKnife':49, 'Toaster':80, 'Spoon':50, 'Knife':49, 'DiningTable':67, 'Bowl':51, 
        #     'Vase':86, 
        #     'TeddyBear':88, 
        # }

        # self.maskrcnn_to_catname = {
        #     81:'sink', 
        #     67:'dining table', 65:'bed', 84:'book', 
        #     77:'cell phone', 70: 'toilet',
        #     85:'clock', 73:'laptop', 62:'chair',
        #     72:'tv', 75:'remote', 64:'potted plant', 
        #     63:'couch', 39:'baseball bat', 43:'tennis racket', 47:'cup', 
        #     53:'apple', 44:'bottle', 78:'microwave', 48:'fork', 82:'refrigerator', 
        #     46:'wine glass', 
        #     49:'knife', 79:'oven', 80:'toaster', 50:'spoon', 67:'dining table', 51:'bowl', 
        #     86:'vase', 
        #     88:'teddy bear', 
        # }

        self.obj_conf_dict = {
            'sink':[], 
            'dining table':[], 'bed':[], 'book':[], 
            'cell phone':[], 
            'clock':[], 'laptop':[], 'chair':[],
            'tv':[], 'remote':[], 'potted plant':[], 
            'couch':[], 'baseball bat':[], 'tennis racket':[], 'cup':[], 
            'apple':[], 'bottle':[], 'microwave':[], 'fork':[], 'refrigerator':[], 
            'wine glass':[], 
            'knife':[], 'oven':[], 'toaster':[], 'spoon':[], 'dining table':[], 'bowl':[], 
            'vase':[], 
            'teddy bear':[], 
        }

        self.data_store = {
            'sink':{}, 
            'dining table':{}, 'bed':{}, 'book':{}, 
            'cell phone':{}, 
            'clock':{}, 'laptop':{}, 'chair':{},
            'tv':{}, 'remote':{}, 'potted plant':{}, 
            'couch':{}, 'baseball bat':{}, 'tennis racket':{}, 'cup':{}, 
            'apple':{}, 'bottle':{}, 'microwave':{}, 'fork':{}, 'refrigerator':{}, 
            'wine glass':{}, 
            'knife':{}, 'oven':{}, 'toaster':{}, 'spoon':{}, 'dining table':{}, 'bowl':{}, 
            'vase':{}, 
            'teddy bear':{}, 
        }

        self.data_store_features = []
        self.feature_obj_ids = []
        self.first_time = True
        self.Softmax = nn.Softmax(dim=0)

        self.action_space = {0: "MoveLeft", 1: "MoveRight", 2: "MoveAhead", 3: "MoveBack", 4: "DoNothing"}
        self.num_actions = len(self.action_space)

        # cfg_det = get_cfg()
        # cfg_det.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # cfg_det.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
        # cfg_det.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # cfg_det.MODEL.DEVICE='cuda'
        # self.cfg_det = cfg_det
        # self.maskrcnn = DefaultPredictor(cfg_det)

        cfg_det = get_cfg()
        cfg_det.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_det.MODEL.MASK_ON = True
        cfg_det.MODEL.CLS_AGNOSTIC_BBOX_REG = True
        cfg_det.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
        cfg_det.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg_det.MODEL.BACKBONE.FREEZE_AT = 0
        cfg_det.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg_det.MODEL.DEVICE='cuda'
        cfg_det.MODEL.WEIGHTS = "/home/sirdome/katefgroup/gsarch/ithor/iThor_Scripts/scripts/checkpoints/model_0069999.pth"
        thing_classes = ['']
        d='train'
        DatasetCatalog.register("ag_val", lambda d=d: val_dataset_function())
        MetadataCatalog.get("ag_val").thing_classes = thing_classes
        cfg_det.DATASETS.TRAIN=('ag_val',)
        self.cfg_det = cfg_det
        self.maskrcnn = DefaultPredictor(cfg_det)

        self.normalize = transforms.Compose([
            transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Initialize vgg
        vgg16 = torchvision.models.vgg16(pretrained=True).double().cuda()
        vgg16.eval()
        print(torch.nn.Sequential(*list(vgg16.features.children())))
        self.vgg_feat_extractor = torch.nn.Sequential(*list(vgg16.features.children())[:-2])
        print(self.vgg_feat_extractor)
        self.vgg_mean = torch.from_numpy(np.array([0.485,0.456,0.406]).reshape(1,3,1,1))
        self.vgg_std = torch.from_numpy(np.array([0.229,0.224,0.225]).reshape(1,3,1,1))

        self.conf_thresh_detect = 0.7 # for initially detecting a low confident object
        self.conf_thresh_init = 0.8 # for after turning head toward object threshold
        self.conf_thresh_end = 0.9 # if reach this then stop getting obs

        self.BATCH_SIZE = 50 # frames (not episodes) - this is approximate - it could be higher 
        # self.percentile = 70
        self.max_iters = 100000
        self.max_frames = 10
        self.val_interval = 10 #10
        self.save_interval = 50

        # self.BATCH_SIZE = 2
        # self.percentile = 70
        # self.max_iters = 100000
        # self.max_frames = 2
        # self.val_interval = 1
        # self.save_interval = 1

        self.small_classes = []
        self.rot_interval = 5.0
        self.radius_max = 3.5 #3 #1.75
        self.radius_min = 1.0 #1.25
        self.num_flat_views = 3
        self.num_any_views = 7
        self.num_views = 25
        self.center_from_mask = False # get object centroid from maskrcnn (True) or gt (False)

        self.obj_per_scene = 5

        mod = 'ds01'

        # self.homepath = f'/home/nel/gsarch/aithor/data/test2'
        self.homepath = '/home/sirdome/katefgroup/gsarch/ithor/data/' + mod
        print(self.homepath)
        if not os.path.exists(self.homepath):
            os.mkdir(self.homepath)
        else:
            val = input("Delete homepath? [y/n]: ")
            if val == 'y':
                import shutil
                shutil.rmtree(self.homepath)
                os.mkdir(self.homepath)
            else:
                print("ENDING")
                assert(False)

        self.log_freq = 1
        self.log_dir = self.homepath +'/..' + '/log_cem/' + mod
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        MAX_QUEUE = 10 # flushes when this amount waiting
        self.writer = SummaryWriter(self.log_dir, max_queue=MAX_QUEUE, flush_secs=60)


        self.W = 256
        self.H = 256

        self.fov = 90

        self.utils = Utils(self.fov, self.W, self.H)
        self.K = self.utils.get_habitat_pix_T_camX(self.fov)
        self.camera_matrix = self.utils.get_camera_matrix(self.W, self.H, self.fov)

        self.controller = Controller(
            scene='FloorPlan30', # will change 
            gridSize=0.25,
            width=self.W,
            height=self.H,
            fieldOfView= self.fov,
            renderObjectImage=True,
            renderDepthImage=True,
            )

        self.init_network()

        self.run_episodes()
    
    def init_network(self):

        input_shape = np.array([3, self.W, self.H])
        
        self.localpnet = LocalPNET(input_shape=input_shape, num_actions=self.num_actions).cuda()

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(params=self.localpnet.parameters(),lr=0.00001)

    def batch_iteration(self,mapnames,BATCH_SIZE):

        batch = {"actions": [], "obs_all": [], "seg_ims": [], "conf_end_change": [], "conf_avg_change": [], "conf_median_change": []}
        iter_idx = 0
        total_loss = torch.tensor(0.0).cuda()
        num_obs = 0
        while True:

            mapname = np.random.choice(mapnames)

            # self.basepath = self.homepath + f"/{mapname}_{episode}"
            # print("BASEPATH: ", self.basepath)

            # # self.basepath = f"/hdd/ayushj/habitat_data/{mapname}_{episode}"
            # if not os.path.exists(self.basepath):
            #     os.mkdir(self.basepath)

            self.controller.reset(scene=mapname)

            total_loss, obs, actions, seg_ims, confs = self.run("train", total_loss)

            if obs is None:
                print("NO EPISODE LOSS.. SKIPPING BATCH INSTANCE")
                continue

            num_obs += len(actions)

            print("Total loss for train batch # ",iter_idx," :",total_loss)

            confs = np.array(confs)
            conf_end_change = confs[-1] - confs[0]
            conf_avg_change = np.mean(np.diff(confs))
            conf_median_change = np.median(np.diff(confs))

            batch["actions"].append(actions) 
            # These are only used for plotting
            batch["obs_all"].append(obs)
            batch["seg_ims"].append(seg_ims)
            batch["conf_end_change"].append(conf_end_change)
            batch["conf_avg_change"].append(conf_avg_change)
            batch["conf_median_change"].append(conf_median_change)

            iter_idx += 1   

            # if len(batch["obs_all"]) == BATCH_SIZE:
            if num_obs >= BATCH_SIZE:
                print("NUM OBS IN BATCH=", num_obs)
                # batch["total_loss"] = total_loss
                print("Total loss for iter: ", total_loss)

                return total_loss, batch, num_obs
                # iter_idx = 0
                # total_loss = torch.tensor(0.0).cuda()
                # batch = {"actions": [], "obs_all": [], "seg_ims": [], "conf_end_change": [], "conf_avg_change": []}
                
    def run_episodes(self):
        self.ep_idx = 0
        # self.objects = []
        
        # get object centers from ground truth for first house
        self.center_from_mask = False
        for episode in range(len(self.mapnames_train)):
            print("STARTING EPISODE ", episode)

            mapname = self.mapnames_train[episode]
            print("MAPNAME=", mapname)

            self.controller.reset(scene=mapname)

            # self.controller.start()
            
            self.basepath = self.homepath + f"/{mapname}_{episode}"
            print("BASEPATH: ", self.basepath)

            # self.basepath = f"/hdd/ayushj/habitat_data/{mapname}_{episode}"
            if not os.path.exists(self.basepath):
                os.mkdir(self.basepath)

            self.run(mode="train")

            self.ep_idx += 1

        self.ep_idx = 1
        self.best_inner_prods = []
        self.pred_ids = []
        self.true_ids = []
        self.pred_catnames = []
        self.true_catnames = []
        self.pred_catnames_all = []
        self.true_catnames_all = []
        self.conf_mats = []
        # self.pred_catnames = []

        # get object centers from detections for next houses
        self.center_from_mask = True
        self.data_store_features_cur_house = []
        self.feature_obj_ids_cur_house = []
        self.first_time_house = True
        for episode in range(len(self.mapnames_test)):
            print("STARTING EPISODE ", episode)

            mapname = self.mapnames_test[episode]
            print("MAPNAME=", mapname)

            self.controller.reset(scene=mapname)

            # self.controller.start()
            
            self.basepath = self.homepath + f"/{mapname}_{episode}"
            print("BASEPATH: ", self.basepath)

            # self.basepath = f"/hdd/ayushj/habitat_data/{mapname}_{episode}"
            if not os.path.exists(self.basepath):
                os.mkdir(self.basepath)

            self.run(mode="test")

            if self.ep_idx % 4 == 0:
                self.data_store_features = np.vstack((self.data_store_features, self.data_store_features_cur_house))
                self.feature_obj_ids.extend(self.feature_obj_ids_cur_house)
                self.data_store_features_cur_house = []
                self.feature_obj_ids_cur_house = []
                self.first_time_house = True

                self.best_inner_prods = np.array(self.best_inner_prods)
                self.pred_ids = np.array(self.pred_ids)
                self.true_ids = np.array(self.true_ids)
                # for i in range(len(self.best_inner_prods)):s

                correct_pred = self.best_inner_prods[self.pred_ids==self.true_ids]
                incorrect_pred = self.best_inner_prods[self.pred_ids!=self.true_ids]
                
                bins = 50
                plt.figure(1)
                plt.clf()
                plt.hist([correct_pred, incorrect_pred], alpha=0.5, histtype='stepfilled', label=['correct', 'incorrect'], bins=bins)
                plt.title(f'testhouse{self.ep_idx//4}')
                plt.xlabel('inner product of nearest neighbor')
                plt.ylabel('Counts')
                plt.legend()
                plt_name = self.homepath + f'/correct_incorrect_testhouse{self.ep_idx//4}.png'
                plt.savefig(plt_name)

                self.unique_classes = []
                self.unique_classes.extend(self.pred_catnames)
                self.unique_classes.extend(self.true_catnames)
                self.unique_classes = list(set(self.unique_classes))

                conf_mat = confusion_matrix(self.pred_catnames, self.true_catnames, labels=self.unique_classes)
                self.conf_mats.append(conf_mat)
                
                plt.figure(1)
                plt.clf()
                df_cm = pd.DataFrame(conf_mat, index = [i for i in self.unique_classes],
                  columns = [i for i in self.unique_classes])
                plt.figure(figsize = (10,7))
                sn.heatmap(df_cm, annot=True)
                plt.xticks(np.arange(len(self.unique_classes))+0.5, self.unique_classes)
                plt.yticks(np.arange(len(self.unique_classes))+0.5, self.unique_classes)
                plt_name = self.homepath + f'/confusion_matrix_testhouse{self.ep_idx//4}.png'
                plt.savefig(plt_name)
                # plt.show()

                self.pred_catnames_all.extend(self.pred_catnames)
                self.true_catnames_all.extend(self.true_catnames)

                self.unique_classes = []
                self.unique_classes.extend(self.pred_catnames_all)
                self.unique_classes.extend(self.true_catnames_all)
                self.unique_classes = list(set(self.unique_classes))
                
                
                self.best_inner_prods = []
                self.pred_ids = []
                self.true_ids = []
                self.true_catnames = []
                self.pred_catnames = []
                self.true_catnames = []

                conf_mat = confusion_matrix(self.pred_catnames_all, self.true_catnames_all, labels=self.unique_classes)
                plt.figure(1)
                plt.clf()
                df_cm = pd.DataFrame(conf_mat, index = [i for i in self.unique_classes],
                  columns = [i for i in self.unique_classes])
                plt.figure(figsize = (10,7))
                sn.heatmap(df_cm, annot=True)
                plt.xticks(np.arange(len(self.unique_classes))+0.5, self.unique_classes)
                plt.yticks(np.arange(len(self.unique_classes))+0.5, self.unique_classes)
                plt_name = self.homepath + f'/confusion_matrix_testhouses_all.png'
                plt.savefig(plt_name)
                

            self.ep_idx += 1


        # replot the end result
        conf_mat = confusion_matrix(self.pred_catnames_all, self.true_catnames_all, labels=self.unique_classes)
        plt.figure(1)
        plt.clf()
        df_cm = pd.DataFrame(conf_mat, index = [i for i in self.unique_classes],
            columns = [i for i in self.unique_classes])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True)
        plt.xticks(np.arange(len(self.unique_classes))+0.5, self.unique_classes)
        plt.yticks(np.arange(len(self.unique_classes))+0.5, self.unique_classes)
        plt.show()

        self.controller.stop()
        time.sleep(1)
    
    def run2(self):
        event = self.controller.step('GetReachablePositions')
        for obj in event.metadata['objects']:
            if obj['objectType'] not in self.objects:
                self.objects.append(obj['objectType'])

    
    def get_detectron_conf_center_obj(self,im, obj_mask, frame=None):
        im = Image.fromarray(im, mode="RGB")
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

        outputs = self.maskrcnn(im)

        pred_masks = outputs['instances'].pred_masks
        pred_scores = outputs['instances'].scores
        pred_classes = outputs['instances'].pred_classes
        pred_boxes = outputs['instances'].pred_boxes.tensor

        len_pad = 5

        W2_low = self.W//2 - len_pad
        W2_high = self.W//2 + len_pad
        H2_low = self.H//2 - len_pad
        H2_high = self.H//2 + len_pad

        if False:

            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg_det.DATASETS.TRAIN[0]), scale=1.0)
            out = v.draw_instance_predictions(outputs['instances'].to("cpu"))
            seg_im = out.get_image()
        
            plt.figure(1)
            plt.clf()
            plt.imshow(seg_im)
            plt_name = self.homepath + f'/seg_all{frame}.png'
            plt.savefig(plt_name)

            seg_im[W2_low:W2_high, H2_low:H2_high,:] = 0.0
            plt.figure(1)
            plt.clf()
            plt.imshow(seg_im)
            plt_name = self.homepath + f'/seg_all_mask{frame}.png'
            plt.savefig(plt_name)

        ind_obj = None
        # max_overlap = 0
        sum_obj_mask = np.sum(obj_mask)
        mask_sum_thresh = 7000
        for idx in range(pred_masks.shape[0]):
            pred_mask_cur = pred_masks[idx].detach().cpu().numpy()
            pred_masks_center = pred_mask_cur[W2_low:W2_high, H2_low:H2_high]
            sum_pred_mask_cur = np.sum(pred_mask_cur)
            # print(torch.sum(pred_masks_center))
            if np.sum(pred_masks_center) > 0:
                if np.abs(sum_pred_mask_cur - sum_obj_mask) < mask_sum_thresh:
                    ind_obj = idx
                    mask_sum_thresh = np.abs(sum_pred_mask_cur - sum_obj_mask)
                # max_overlap = torch.sum(pred_masks_center)
        if ind_obj is None:
            print("RETURNING NONE")
            return None, None, None, None, None

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg_det.DATASETS.TRAIN[0]), scale=1.0)
        out = v.draw_instance_predictions(outputs['instances'][ind_obj].to("cpu"))
        seg_im = out.get_image()

        if False:
            plt.figure(1)
            plt.clf()
            plt.imshow(seg_im)
            plt_name = self.homepath + f'/seg{frame}.png'
            plt.savefig(plt_name)

        # print("OBJ CLASS ID=", int(pred_classes[ind_obj].detach().cpu().numpy()))
        # pred_boxes = outputs['instances'].pred_boxes.tensor
        # pred_classes = outputs['instances'].pred_classes
        # pred_scores = outputs['instances'].scores
        obj_score = float(pred_scores[ind_obj].detach().cpu().numpy())
        obj_pred_classes = int(pred_classes[ind_obj].detach().cpu().numpy())
        obj_pred_mask = pred_masks[ind_obj].detach().cpu().numpy()
        obj_pred_boxes = pred_boxes[ind_obj].detach().cpu().numpy()


        return obj_score, obj_pred_classes, obj_pred_mask, obj_pred_boxes, seg_im            

            
    def detect_object_centroid(self, im, event):

        im = Image.fromarray(im, mode="RGB")
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

        outputs = self.maskrcnn(im)

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg_det.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs['instances'].to("cpu"))
        seg_im = out.get_image()

        if False:
            plt.figure(1)
            plt.clf()
            plt.imshow(seg_im)
            plt.show()
            # plt_name = self.homepath + '/seg_init.png'
            # plt.savefig(plt_name)

        pred_masks = outputs['instances'].pred_masks
        pred_boxes = outputs['instances'].pred_boxes.tensor
        pred_classes = outputs['instances'].pred_classes
        pred_scores = outputs['instances'].scores

        obj_catids = []
        obj_scores = []
        obj_masks = []
        obj_boxes = []
        for segs in range(len(pred_masks)):
            if pred_scores[segs] <= self.conf_thresh_detect:
                obj_catids.append(pred_classes[segs].item())
                obj_scores.append(pred_scores[segs].item())
                obj_masks.append(pred_masks[segs])
                obj_boxes.append(pred_boxes[segs])

        eulers_xyz_rad = np.radians(np.array([event.metadata['agent']['cameraHorizon'], event.metadata['agent']['rotation']['y'], 0.0]))

        rx = eulers_xyz_rad[0]
        ry = eulers_xyz_rad[1]
        rz = eulers_xyz_rad[2]
        rotation_ = self.utils.eul2rotm(-rx, -ry, rz)

        translation_ = np.array(list(event.metadata['agent']['position'].values())) + np.array([0.0, 0.675, 0.0])
        # need to invert since z is positive here by convention
        translation_[2] =  -translation_[2]

        T_world_cam = np.eye(4)
        T_world_cam[0:3,0:3] =  rotation_
        T_world_cam[0:3,3] = translation_

        if not obj_masks:
            return None, None, None, None
        elif self.center_from_mask: 

            # want an object not on the edges of the image
            sum_interior = 0
            xyz_obj_mid = None
            while True:
                if len(obj_masks)==0:
                    return None, None, None, None
                random_int = np.random.randint(low=0, high=len(obj_masks))
                obj_mask_focus = obj_masks.pop(random_int)
                bbox = obj_boxes.pop(random_int)
                print("OBJECT ID INIT=", obj_catids.pop(random_int))
                sum_interior = torch.sum(obj_mask_focus[50:self.W-50, 50:self.H-50])
                if sum_interior < 500 or sum_interior > self.W*self.H*0.3:
                    continue # exclude too small objects or too large objects
                
                obj_mask_focus = obj_mask_focus.detach().cpu().numpy()
                pixel_locs_obj = np.where(obj_mask_focus)
                x_mid = np.round(np.median(pixel_locs_obj[1])/self.W, 4)
                y_mid = np.round(np.median(pixel_locs_obj[0])/self.H, 4)

                if False:
                    plt.figure(1)
                    plt.clf()
                    plt.imshow(obj_mask_focus)
                    plt.plot(np.median(pixel_locs_obj[1]), np.median(pixel_locs_obj[0]), 'x')
                    plt_name = self.homepath + '/seg_mask.png'
                    plt.savefig(plt_name)

                
                event = self.controller.step('TouchThenApplyForce', x=x_mid, y=y_mid, handDistance = 1000000.0, direction=dict(x=0.0, y=0.0, z=0.0), moveMagnitude = 0.0)
                obj_focus_id = event.metadata['actionReturn']['objectId']

                depth = event.depth_frame

                xs, ys = np.meshgrid(np.linspace(-1*256/2.,1*256/2.,256), np.linspace(1*256/2.,-1*256/2., 256))
                depth = depth.reshape(1,256,256)
                xs = xs.reshape(1,256,256)
                ys = ys.reshape(1,256,256)

                xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
                xys = xys.reshape(4, -1)
                xy_c0 = np.matmul(np.linalg.inv(self.K), xys)
                xyz = xy_c0.T[:,:3].reshape(256,256,3)
                xyz_obj_masked = xyz[obj_mask_focus]

                xyz_obj_masked = np.matmul(rotation_, xyz_obj_masked.T) + translation_.reshape(3,1)
                xyz_obj_mid = np.mean(xyz_obj_masked, axis=1)

                # y axis is flipped
                xyz_obj_mid[2] = -xyz_obj_mid[2]

                object_type = None
                for o in event.metadata['objects']:
                    if o['objectId'] == obj_focus_id:
                        if o['objectType'] not in self.include_classes:
                            continue
                        object_type = o['objectType']

                
                if xyz_obj_mid is not None:
                    if object_type is not None:
                        break

            if False:
                plt.figure(1)
                plt.clf()
                plt.imshow(seg_im)

                plt.figure(2)
                plt.clf()
                plt.imshow(obj_mask_focus)

                plt.show()
                # plt_name = self.homepath + '/seg_init.png'
                # plt.savefig(plt_name)

            
        else:

            # want an object not on the edges of the image
            
            sum_interior = 0
            while True:
                if len(obj_masks)==0:
                    return None, None, None
                random_int = np.random.randint(low=0, high=len(obj_masks))
                obj_mask_focus = obj_masks.pop(random_int)
                # print("OBJECT ID INIT=", obj_catids[random_int])
                sum_interior = torch.sum(obj_mask_focus[50:self.W-50, 50:self.H-50])
                if sum_interior < 500:
                    continue # exclude too small objects


                pixel_locs_obj = np.where(obj_mask_focus.cpu().numpy())
                x_mid = np.round(np.median(pixel_locs_obj[1])/self.W, 4)
                y_mid = np.round(np.median(pixel_locs_obj[0])/self.H, 4)

                if False:
                    plt.figure(1)
                    plt.clf()
                    plt.imshow(obj_mask_focus)
                    plt.plot(np.median(pixel_locs_obj[1]), np.median(pixel_locs_obj[0]), 'x')
                    plt_name = self.homepath + '/seg_mask.png'
                    plt.savefig(plt_name)

                
                event = self.controller.step('TouchThenApplyForce', x=x_mid, y=y_mid, handDistance = 1000000.0, direction=dict(x=0.0, y=0.0, z=0.0), moveMagnitude = 0.0)
                obj_focus_id = event.metadata['actionReturn']['objectId']

                xyz_obj_mid = None
                for o in event.metadata['objects']:
                    if o['objectId'] == obj_focus_id:
                        if o['objectType'] not in self.include_classes:
                            continue
                        xyz_obj_mid = np.array(list(o['axisAlignedBoundingBox']['center'].values()))
                        object_type = o['objectType']
                
                if xyz_obj_mid is not None:
                    break

        print("MIDPOINT=", xyz_obj_mid)
        return xyz_obj_mid, obj_mask_focus, object_type, bbox


    def run(self, mode=None, total_loss=None, summ_writer=None):
        
        event = self.controller.step('GetReachablePositions')
        if not event.metadata['reachablePositions']:
            # Different versions this is empty/full
            event = self.controller.step(action='MoveAhead')
        self.nav_pts = event.metadata['reachablePositions']
        self.nav_pts = np.array([list(d.values()) for d in self.nav_pts])
        # objects = np.random.choice(event.metadata['objects'], self.obj_per_scene, replace=False)
        objects = event.metadata['objects']
        objects_inds = np.arange(len(event.metadata['objects']))
        np.random.shuffle(objects_inds)

        # objects = np.random.shuffle(event.metadata['objects'])
        # for obj in event.metadata['objects']: #objects:
        #     print(obj['name'])
        # objects = objects[0]
        successes = 0
        # meta_obj_idx = 0
        num_obs = 0
        # while successes < self.obj_per_scene and meta_obj_idx <= len(event.metadata['objects']) - 1: 
        for obj in objects:
            # if meta_obj_idx > len(event.metadata['objects']) - 1:
            #     print("OUT OF OBJECT... RETURNING")
            #     return total_loss, None, None, None, None
                
            # obj = objects[objects_inds[meta_obj_idx]]
            # meta_obj_idx += 1
            print("Center object is ", obj['objectType'])
            # if obj['name'] in ['Microwave_b200e0bc']:
            #     print(obj['name'])
            # else:
            #     continue
            # print(obj['name'])

            if obj['objectType'] not in self.include_classes:
                print("Continuing... Invalid Object")
                continue

            if mode == "train":
                # Calculate distance to object center
                obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))

                obj_center = np.expand_dims(obj_center, axis=0)
            else:
                # For testing, we'll spawn with objects in view and then randomly get centroid of detected object in view

                # Calculate distance to object center
                obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))
                            
                obj_center = np.expand_dims(obj_center, axis=0)
                distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))

                # Get points with r_min < dist < r_max
                valid_pts = self.nav_pts[np.where((distances > self.radius_min)*(distances<self.radius_max))]

                # add height from center of agent to camera
                rand_pos_int = np.random.randint(low=0, high=valid_pts.shape[0])
                pos_s = valid_pts[rand_pos_int]
                pos_s[1] = pos_s[1] + 0.675

                turn_yaw, turn_pitch = self.utils.get_rotation_to_obj(obj_center, pos_s)
                event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1], z=pos_s[2], rotation=dict(x=0.0, y=int(turn_yaw), z=0.0), horizon=int(turn_pitch))
                rgb = event.frame

                # get object center of a low confidence object
                obj_center, obj_mask, object_type, object_bbox = self.detect_object_centroid(rgb, event)

                if obj_center is None:
                    print("NO Center found... SKIPPING...")
                    continue

            distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))

            # Get points with r_min < dist < r_max
            valid_pts = self.nav_pts[np.where((distances > self.radius_min)*(distances<self.radius_max))]

            # Bin points based on angles [vertical_angle (10 deg/bin), horizontal_angle (10 deg/bin)]
            valid_pts_shift = valid_pts - obj_center

            dz = valid_pts_shift[:,2]
            dx = valid_pts_shift[:,0]
            dy = valid_pts_shift[:,1]

            # Get yaw for binning 
            valid_yaw = np.degrees(np.arctan2(dz,dx))

            
            if mode == "train":
                nbins = 10 #20
            else:
                nbins = 5
            bins = np.linspace(-180, 180, nbins+1)
            bin_yaw = np.digitize(valid_yaw, bins)

            num_valid_bins = np.unique(bin_yaw).size            

            if False:
                import matplotlib.cm as cm
                colors = iter(cm.rainbow(np.linspace(0, 1, nbins)))
                plt.figure(2)
                plt.clf()
                print(np.unique(bin_yaw))
                for bi in range(nbins):
                    cur_bi = np.where(bin_yaw==(bi+1))
                    points = valid_pts[cur_bi]
                    x_sample = points[:,0]
                    z_sample = points[:,2]
                    plt.plot(z_sample, x_sample, 'o', color = next(colors))
                plt.plot(self.nav_pts[:,2], self.nav_pts[:,0], 'x', color='red')
                plt.plot(obj_center[:,2], obj_center[:,0], 'x', color = 'black')
                plt_name = '/home/nel/gsarch/aithor/data/valid.png'
                plt.savefig(plt_name)

            if num_valid_bins == 0:
                continue
            
            if mode == "train":
                spawns_per_bin = 3 #20
            else:
                spawns_per_bin = 1 #int(self.num_views / num_valid_bins) + 2
            # print(f'spawns_per_bin: {spawns_per_bin}')

            action = "do_nothing"
            episodes = []
            valid_pts_selected = []
            camXs_T_camX0_4x4 = []
            camX0_T_camXs_4x4 = []
            origin_T_camXs = []
            origin_T_camXs_t = []
            cnt = 0
            for b in range(nbins):
                
                # get all angle indices in the current bin range
                inds_bin_cur = np.where(bin_yaw==(b+1)) # bins start 1 so need +1
                inds_bin_cur = list(inds_bin_cur[0])
                if len(inds_bin_cur) == 0:
                    continue

                for s in range(spawns_per_bin):

                    observations = {}
                    
                    if len(inds_bin_cur) == 0:
                        continue
                    
                    rand_ind = np.random.randint(0, len(inds_bin_cur))
                    s_ind = inds_bin_cur.pop(rand_ind)

                    pos_s = valid_pts[s_ind]
                    valid_pts_selected.append(pos_s)

                    # add height from center of agent to camera
                    pos_s[1] = pos_s[1] + 0.675

                    turn_yaw, turn_pitch = self.utils.get_rotation_to_obj(obj_center, pos_s)

                    event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1], z=pos_s[2], rotation=dict(x=0.0, y=int(turn_yaw), z=0.0), horizon=int(turn_pitch))

                    rgb = event.frame

                    if mode=="train":
                        object_id = obj['objectId']   
                        instance_detections2D = event.instance_detections2D                 
                        if object_id not in instance_detections2D:
                            print("NOT in instance detections 2D.. continuing")
                            continue
                        obj_instance_detection2D = instance_detections2D[object_id] # [start_x, start_y, end_x, end_y]
                    else:
                        obj_score, obj_pred_classes, obj_mask_new, object_bbox, seg_im = self.get_detectron_conf_center_obj(rgb, obj_mask, frame=None)
                        if obj_mask_new is None:
                            print("object mask is none.. continuing")
                            continue
                        obj_mask = obj_mask_new
                        obj_instance_detection2D = object_bbox # predited box [start_x, start_y, end_x, end_y]
                    
                    # print("obj_instance_detection2D=", obj_instance_detection2D)

                    max_len = np.max(np.array([obj_instance_detection2D[2] - obj_instance_detection2D[0], obj_instance_detection2D[3] - obj_instance_detection2D[1]]))
                    pad_len = max_len//12

                    if pad_len==0:
                        print("pad len 0.. continuing")
                        continue

                    x_center = (obj_instance_detection2D[3] + obj_instance_detection2D[1]) // 2
                    x_low = int(x_center-max_len-pad_len)
                    if x_low < 0:
                        x_low = 0
                    x_high = int(x_center+max_len+pad_len) #x_low + max_len + 2*pad_len
                    if x_high > self.W:
                        x_high = self.W

                    y_center = (obj_instance_detection2D[2] + obj_instance_detection2D[0]) // 2
                    y_low = int(y_center-max_len-pad_len)#-pad_len
                    if y_low < 0:
                        y_low = 0
                    y_high = int(y_center+max_len+pad_len) #y_low + max_len + 2*pad_len
                    if y_high > self.H:
                        y_high = self.H

                    rgb_crop = rgb[x_low:x_high, y_low:y_high,:]

                    rgb_crop = Image.fromarray(rgb_crop)

                    normalize_cropped_rgb = self.normalize(rgb_crop).unsqueeze(0).double().cuda()

                    obj_features = self.vgg_feat_extractor(normalize_cropped_rgb).view((512, -1))

                    obj_features = obj_features.detach().cpu().numpy()

                    # pca = PCA(n_components=10)
                    # obj_features = pca.fit_transform(obj_features.T).flatten()

                    # obj_features = torch.from_numpy(obj_features).view(-1).cuda()
                    obj_features = obj_features.flatten()

                    if mode=="train":
                        if self.first_time:
                            self.first_time = False
                            self.data_store_features = obj_features
                            # self.data_store_features = self.data_store_features.cuda()
                            self.feature_obj_ids.append(self.ithor_to_maskrcnn[obj['objectType']])
                        else:
                            # self.data_store_features = torch.vstack((self.data_store_features, obj_features))
                            self.data_store_features = np.vstack((self.data_store_features, obj_features))
                            self.feature_obj_ids.append(self.ithor_to_maskrcnn[obj['objectType']])

                    elif mode=="test":

                        # obj_features = obj_features.unsqueeze(0)

                        # inner_prod = torch.abs(torch.mm(obj_features, self.data_store_features.T)).squeeze()

                        # inner_prod = inner_prod.detach().cpu().numpy()

                        # dist = np.squeeze(np.abs(np.matmul(obj_features, self.data_store_features.transpose())))

                        dist = np.linalg.norm(self.data_store_features-obj_features, axis=1)

                        k = 10

                        ind_knn = list(np.argsort(dist)[:k])

                        dist_knn = np.sort(dist)[:k]
                        dist_knn_norm = list(self.Softmax(torch.from_numpy(-dist_knn)).numpy())

                        match_knn_id = [self.feature_obj_ids[i] for i in ind_knn] 

                        # for i in range(1, len(match_knn_id)):

                        # add softmax values from the same class (probably a really complex way of doing this)
                        idx = 0
                        dist_knn_norm_add = []
                        match_knn_id_add = []
                        while True:
                            if not match_knn_id:
                                break
                            match_knn_cur = match_knn_id.pop(0)
                            dist_knn_norm_cur = dist_knn_norm.pop(0)
                            match_knn_id_add.append(match_knn_cur)
                            idxs_ = []
                            for i in range(len(match_knn_id)):
                                if match_knn_id[i] == match_knn_cur:
                                    dist_knn_norm_cur += dist_knn_norm[i]
                                    # match_knn_id_.pop(i)
                                else:
                                    idxs_.append(i)
                            match_knn_id = [match_knn_id[idx] for idx in idxs_]
                            dist_knn_norm = [dist_knn_norm[idx] for idx in idxs_]
                            dist_knn_norm_add.append(dist_knn_norm_cur)

                        dist_knn_norm_add = np.array(dist_knn_norm_add)

                        dist_knn_argmax = np.argmax(dist_knn_norm_add)

                        match_nn_id = match_knn_id_add[dist_knn_argmax] #self.feature_obj_ids[ind_nn]

                        match_nn_catname = self.maskrcnn_to_ithor[match_nn_id]

                        self.best_inner_prods.append(dist_knn_norm_add[dist_knn_argmax])
                        self.pred_ids.append(match_nn_id)
                        # self.pred_catnames.append(match_nn_catname)
                        self.true_ids.append(self.ithor_to_maskrcnn[object_type])
                        self.pred_catnames.append(match_nn_catname)
                        self.true_catnames.append(object_type)

                        print(match_nn_catname)

                        if self.first_time_house:
                            self.first_time_house = False
                            self.data_store_features_cur_house = obj_features
                            self.feature_obj_ids_cur_house.append(self.ithor_to_maskrcnn[object_type])
                        else:
                            self.data_store_features_cur_house = np.vstack((self.data_store_features_cur_house, obj_features))
                            self.feature_obj_ids_cur_house.append(self.ithor_to_maskrcnn[object_type])

                        

                        if False:
                            normalize_cropped_rgb = np.transpose(normalize_cropped_rgb.squeeze(0).detach().cpu().numpy(), (1,2,0))
                            plt.figure(1)
                            plt.clf()
                            plt.imshow(normalize_cropped_rgb)
                            # plt_name = self.homepath + '/seg_init.png'
                            plt.figure(2)
                            plt.clf()
                            plt.imshow(rgb)
                            print(obj_instance_detection2D[0], obj_instance_detection2D[2], obj_instance_detection2D[1], obj_instance_detection2D[3])
                            plt.plot([obj_instance_detection2D[0], obj_instance_detection2D[2]], [obj_instance_detection2D[1], obj_instance_detection2D[3]], 'x', color='red')
                            plt.show()

                            plt.figure(3)
                            plt.clf()
                            plt.imshow(np.array(rgb_crop))
                            plt.show()

                    # except Exception as e:
                    #     print(e)
                    #     st()

                    # eulers_xyz_rad = np.radians(np.array([event.metadata['agent']['cameraHorizon'], event.metadata['agent']['rotation']['y'], 0.0]))

                    # rx = eulers_xyz_rad[0]
                    # ry = eulers_xyz_rad[1]
                    # rz = eulers_xyz_rad[2]
                    # rotation_r_matrix = self.utils.eul2rotm(-rx, -ry, rz)

                    # agent_position = np.array(list(event.metadata['agent']['position'].values())) + np.array([0.0, 0.675, 0.0])
                    # # need to invert since z is positive here by convention
                    # agent_position[2] =  -agent_position[2]


                    # observations["positions"] = agent_position

                    # observations["rotations"] = rotation_r_matrix

                    # # rt_4x4 = np.eye(4)
                    # # rt_4x4[0:3,0:3] = observations["rotations"]
                    # # rt_4x4[0:3,3] = observations["positions"]
                    # # rt_4x4_inv = self.safe_inverse_single(rt_4x4)
                    # # r, t = self.split_rt_single(rt_4x4_inv)

                    # # observations["positions"] = r

                    # # observations["positions"] = t

                    # # observations["rotations_euler"] = np.array([rx, ry, rz]) #rotation_r.as_euler('xyz', degrees=True)

                    # observations["color_sensor"] = rgb
                    # observations["depth_sensor"] = event.depth_frame
                    # observations["semantic_sensor"] = event.instance_segmentation_frame

                    # if False:
                    #     plt.imshow(rgb)
                    #     plt_name = f'/home/nel/gsarch/aithor/data/test/img_true{s}{b}.png'
                    #     plt.savefig(plt_name)

                    # # print("Processed image #", cnt, " for object ", obj['objectType'])

                    # semantic = event.instance_segmentation_frame
                    # object_id_to_color = event.object_id_to_color
                    # color_to_object_id = event.color_to_object_id

                    # obj_ids = np.unique(semantic.reshape(-1, semantic.shape[2]), axis=0)

                    # instance_masks = event.instance_masks
                    # instance_detections2D = event.instance_detections2D
                    
                    # obj_metadata_IDs = []
                    # for obj_m in event.metadata['objects']: #objects:
                    #     obj_metadata_IDs.append(obj_m['objectId'])
                    
                    # object_list = []
                    # for obj_idx in range(obj_ids.shape[0]):
                    #     try:
                    #         obj_color = tuple(obj_ids[obj_idx])
                    #         object_id = color_to_object_id[obj_color]
                    #     except:
                    #         # print("Skipping ", object_id)
                    #         continue

                    #     if object_id not in obj_metadata_IDs:
                    #         # print("Skipping ", object_id)
                    #         continue

                    #     obj_meta_index = obj_metadata_IDs.index(object_id)
                    #     obj_meta = event.metadata['objects'][obj_meta_index]
                    #     obj_category_name = obj_meta['objectType']
                        
                    #     # continue if not visible or not in include classes
                    #     if obj_category_name not in self.include_classes or not obj_meta['visible']:
                    #         continue

                    #     obj_instance_mask = instance_masks[object_id]
                    #     obj_instance_detection2D = instance_detections2D[object_id] # [start_x, start_y, end_x, end_y]
                    #     obj_instance_detection2D = np.array([obj_instance_detection2D[1], obj_instance_detection2D[0], obj_instance_detection2D[3], obj_instance_detection2D[2]])  # ymin, xmin, ymax, xmax

                    #     if False:
                    #         print(object_id)
                    #         plt.imshow(obj_instance_mask)
                    #         plt_name = f'/home/nel/gsarch/aithor/data/test/img_mask{s}.png'
                    #         plt.savefig(plt_name)

                    #     obj_center_axisAligned = np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))
                    #     obj_center_axisAligned[2] = -obj_center_axisAligned[2]
                    #     obj_size_axisAligned = np.array(list(obj_meta['axisAlignedBoundingBox']['size'].values()))
                        
                    #     # print(obj_category_name)

                    #     if self.verbose: 
                    #         print("Saved class name is : ", obj_category_name)

                    #     obj_data = {'instance_id': object_id, 'category_id': object_id, 'category_name': obj_category_name,
                    #                     'bbox_center': obj_center_axisAligned, 'bbox_size': obj_size_axisAligned,
                    #                         'mask_2d': obj_instance_mask, 'box_2d': obj_instance_detection2D}
                    #     # object_list.append(obj_instance)
                    #     object_list.append(obj_data)
                    
                    # observations["object_list"] = object_list

                    # # check if object visible (make sure agent is not behind a wall)
                    # obj_id = obj['objectId']
                    # obj_id_to_color = object_id_to_color[obj_id]
                    # # if np.sum(obj_ids==object_id_to_color[obj_id]) > 0:
                    # if self.verbose:
                    #     print("episode is valid......")
                    # episodes.append(observations)

if __name__ == '__main__':
    # startx()
    Ai2Thor()




