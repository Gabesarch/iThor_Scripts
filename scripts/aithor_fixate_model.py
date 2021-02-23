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
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import Image
from tensorboardX import SummaryWriter

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

        # these are all map names
        a = np.arange(1, 30)
        b = np.arange(201, 231)
        c = np.arange(301, 331)
        d = np.arange(401, 431)
        abcd = np.hstack((a,b,c,d))
        mapnames = []
        for i in list(abcd):
            mapname = 'FloorPlan' + str(i)
            mapnames.append(mapname)

        train_len = int(0.9 * len(mapnames))

        random.shuffle(mapnames)
        self.mapnames_train = mapnames[:train_len]
        self.mapnames_val = mapnames[train_len:]
        # self.num_episodes = len(self.mapnames)   

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

        self.include_classes_final = [
            'Sink', 
            'Toilet', 'Bed', 'Book', 
            'CellPhone', 
            'AlarmClock', 'Laptop', 'Chair',
            'Television', 'RemoteControl', 'HousePlant', 
            'Ottoman', 'ArmChair', 'Sofa', 'BaseballBat', 'TennisRacket', 'Mug', 
            'Apple', 'Bottle', 'Microwave', 'Fork', 'Fridge', 
            'WineBottle', 'Cup', 
            'ButterKnife', 'Toaster', 'Spoon', 'Knife', 'DiningTable', 'Bowl', 
            'Vase', 
            'TeddyBear', 'StoveKnob', 'StoveBurner',
            ]

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

        self.action_space = {0: "MoveLeft", 1: "MoveRight", 2: "MoveAhead", 3: "MoveBack", 4: "DoNothing"}
        self.num_actions = len(self.action_space)

        cfg_det = get_cfg()
        cfg_det.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_det.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
        cfg_det.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg_det.MODEL.DEVICE='cuda'
        self.cfg_det = cfg_det
        self.maskrcnn = DefaultPredictor(cfg_det)

        self.conf_thresh_detect = 0.7 # for initially detecting a low confident object
        self.conf_thresh_init = 0.8 # for after turning head toward object threshold
        self.conf_thresh_end = 0.9 # if reach this then stop getting obs

        self.BATCH_SIZE = 50 #50 # frames (not episodes) - this is approximate - it could be higher 
        # self.percentile = 70
        self.max_iters = 100000
        self.max_frames = 10
        self.val_interval = 10 #10 #10
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

        mod = 'conf05'

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
                
    # def elite_batch(self,batch,percentile):

    #     rewards = np.array(batch["rewards"])
    #     obs = batch["obs"]
    #     actions = batch["actions"]

    #     rewards_mean = float(np.mean(rewards))
        
    #     rewards_boundary = np.percentile(rewards,percentile)

    #     print("Reward boundary: ", rewards_boundary)

    #     rewards_mean = float(np.mean(rewards))

    #     training_obs = []
    #     training_actions = []

    #     for idx in range(rewards.shape[0]):
    #         reward_idx = rewards[idx]
    #         if reward_idx < rewards_boundary:
    #             continue

    #         training_obs.extend(obs[idx])
    #         training_actions.extend(actions[idx])

    #     obs_tensor = torch.FloatTensor(training_obs).permute(0, 3, 1, 2).cuda()
    #     act_tensor = torch.LongTensor(training_actions).cuda()

    #     return obs_tensor, act_tensor, rewards_mean, rewards_boundary
    
    def run_val(self, mapnames, BATCH_SIZE, summ_writer=None):
        # run validation every x steps

        # batch = {"rewards": [], "obs": [], "actions": []}
        episode_rewards = 0.0
        seg_ims_batch = []
        obs_ims_batch = []

        iter_idx = 0
        while True:

            mapname = np.random.choice(mapnames)
        
            self.controller.reset(scene=mapname)

            _, obs, actions, seg_ims, confs = self.run("val", None)

            # self.controller.stop()
            # time.sleep(1)

            if obs is None:
                print("NO EPISODE REWARDS.. SKIPPING BATCH INSTANCE")
                continue

            seg_ims_batch.append(seg_ims)
            obs_ims_batch.append(obs)
            
            confs = np.array(confs)
            conf_end_change = confs[-1] - confs[0]
            conf_avg_change = np.mean(np.diff(confs))
            conf_median_change = np.median(np.diff(confs))

            print("val confidence change (end-start):", conf_end_change)
            print("val average confidence difference between frames", conf_avg_change)

            # batch["obs"].append(obs[1:]) # first obs is initial pos (for plotting)
            # batch["actions"].append(actions) 

            iter_idx += 1   

            if len(obs_ims_batch) == 1: # only one for val
        
                try:
                    if summ_writer is not None:
                        name = 'inputs_val/rgbs_original'
                        self.summ_writer.summ_imgs_aithor(name,obs_ims_batch, self.W, self.H, self.max_frames)
                        name = 'inputs_val/rgbs_maskrcnn'
                        self.summ_writer.summ_imgs_aithor(name,seg_ims_batch, self.W, self.H, self.max_frames)

                except:
                    print("PLOTTING DIDNT WORK")
                    pass

                break
        
        return conf_end_change, conf_avg_change, conf_median_change

    def run_episodes(self):

        iteration = 0
        while True:
            
            iteration += 1
            print("ITERATION #", iteration)

            self.summ_writer = utils.improc.Summ_writer(
                writer=self.writer,
                global_step=iteration,
                log_freq=self.log_freq,
                fps=8,
                just_gif=True)

            total_loss, batch, num_obs = self.batch_iteration(self.mapnames_train,self.BATCH_SIZE)

            self.optimizer.zero_grad()

            total_loss.backward()

            self.optimizer.step()

            if iteration >= self.max_iters:
                print("MAX ITERS REACHED")
                self.writer.close()
                break

            if iteration % self.val_interval == 0:
                conf_end_change, conf_avg_change, conf_median_change = self.run_val(self.mapnames_val, self.BATCH_SIZE, self.summ_writer)
                if self.plot_loss:
                    self.summ_writer.summ_scalar('val_conf_end_change', conf_end_change)
                    self.summ_writer.summ_scalar('val_conf_avg_change', conf_avg_change)
                    self.summ_writer.summ_scalar('val_conf_median_change', conf_median_change)

            if iteration % self.save_interval == 0:
                PATH = self.homepath + f'/checkpoint{iteration}.tar'
                torch.save(self.localpnet.state_dict(), PATH)
            
            if self.plot_loss:
                conf_end_change_t = np.mean(np.array(batch["conf_end_change"]))
                conf_avg_change_t = np.mean(np.array(batch["conf_avg_change"]))
                conf_median_change_t = np.mean(batch["conf_median_change"])
                self.summ_writer.summ_scalar('train_conf_end_change_batchavg', conf_end_change_t)
                self.summ_writer.summ_scalar('train_conf_avg_change_batchavg', conf_avg_change_t)
                self.summ_writer.summ_scalar('train_conf_median_change_batchavg', conf_median_change_t)
                self.summ_writer.summ_scalar('total_loss', total_loss)
            
            ## PLOTTING #############
            try:
                summ_writer = self.summ_writer
                if summ_writer is not None and (iteration % self.val_interval == 0):
                    obs_ims_batch = batch["obs_all"]
                    seg_ims_batch = batch["seg_ims"]

                    name = 'inputs_train/rgbs_original'
                    self.summ_writer.summ_imgs_aithor(name,obs_ims_batch, self.W, self.H, self.max_frames)
                    name = 'inputs_train/rgbs_maskrcnn'
                    self.summ_writer.summ_imgs_aithor(name,seg_ims_batch, self.W, self.H, self.max_frames)
            except:
                print("PLOTTING DIDNT WORK")
                pass
                
            self.writer.close() # close tensorboard to flush

        self.controller.stop()
        time.sleep(10)
    
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
            return None, None, None, None

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


        return obj_score, obj_pred_classes, obj_pred_mask, seg_im            

            
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
            plt_name = self.homepath + '/seg_init.png'
            plt.savefig(plt_name)

        pred_masks = outputs['instances'].pred_masks
        pred_boxes = outputs['instances'].pred_boxes.tensor
        pred_classes = outputs['instances'].pred_classes
        pred_scores = outputs['instances'].scores

        obj_catids = []
        obj_scores = []
        obj_masks = []
        for segs in range(len(pred_masks)):
            if pred_scores[segs] <= self.conf_thresh_detect:
                obj_catids.append(pred_classes[segs].item())
                obj_scores.append(pred_scores[segs].item())
                obj_masks.append(pred_masks[segs])

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
            return None, None
        elif self.center_from_mask: 

            # want an object not on the edges of the image
            sum_interior = 0
            while sum_interior==0:
                if len(obj_masks)==0:
                    return None, None
                random_int = np.random.randint(low=0, high=len(obj_masks))
                obj_mask_focus = obj_masks.pop(random_int)
                print("OBJECT ID INIT=", obj_catids[random_int])
                sum_interior = torch.sum(obj_mask_focus[50:self.W-50, 50:self.H-50])

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

            xyz_obj_mid[2] = -xyz_obj_mid[2]
        else:

            # want an object not on the edges of the image
            sum_interior = 0
            while True:
                if len(obj_masks)==0:
                    return None, None
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
                        if o['objectType'] not in self.include_classes_final:
                            continue
                        xyz_obj_mid = np.array(list(o['axisAlignedBoundingBox']['center'].values()))
                
                if xyz_obj_mid is not None:
                    break

        print("MIDPOINT=", xyz_obj_mid)
        return xyz_obj_mid, obj_mask_focus  


    def run(self, mode, total_loss, summ_writer=None):
        
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
        meta_obj_idx = 0
        num_obs = 0
        while True: #successes < self.obj_per_scene and meta_obj_idx <= len(event.metadata['objects']) - 1: 
            if meta_obj_idx > len(event.metadata['objects']) - 1:
                print("OUT OF OBJECT... RETURNING")
                return total_loss, None, None, None, None
                
            obj = objects[objects_inds[meta_obj_idx]]
            meta_obj_idx += 1
            print("Center object is ", obj['objectType'])
            # if obj['name'] in ['Microwave_b200e0bc']:
            #     print(obj['name'])
            # else:
            #     continue
            # print(obj['name'])

            if obj['objectType'] not in self.include_classes:
                print("Continuing... Invalid Object")
                continue
            
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
            obj_center, obj_mask = self.detect_object_centroid(rgb, event)

            if obj_center is None:
                print("NO LOW CONFIDENCE OBJECTS... SKIPPING...")
                continue

            # initialize object in center of FOV
            turn_yaw, turn_pitch = self.utils.get_rotation_to_obj(obj_center, pos_s)
            if mode=="train":
                pos_s_prev = pos_s
                turn_yaw_prev = turn_yaw
                turn_pitch_prev = turn_pitch
            event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1], z=pos_s[2], rotation=dict(x=0.0, y=int(turn_yaw), z=0.0), horizon=int(turn_pitch))
            rgb = event.frame
            seg_ims = []
            obs = []
            init_conf, obj_pred_classes, obj_mask, seg_im = self.get_detectron_conf_center_obj(rgb, obj_mask.detach().cpu().numpy())
            if init_conf is None:
                print("Nothing detected in the center... SKIPPING")
                continue
            conf_cur = init_conf
            conf_prev = init_conf
            if init_conf > self.conf_thresh_init:
                print("HIGH INITIAL CONFIDENCE... SKIPPING...")
                continue
            seg_ims.append(seg_im)
            obs.append(rgb)
            
            actions = []
            confs = []
            confs.append(conf_cur)
            episode_rewards = 0.0
            frame = 0
            while True:

                rgb_tensor = torch.FloatTensor([rgb]).permute(0, 3, 1, 2).cuda()
                
                if mode=="train":
                    torch.set_grad_enabled(True)
                    action_ind, act_proba = self.localpnet(rgb_tensor)
                elif mode=="val":
                    with torch.no_grad():
                        action_ind, act_proba = self.localpnet(rgb_tensor)

                # action_ind, act_proba = actions_probability.data.cpu().numpy()[0]

                # action_ind = np.random.choice(len(act_proba),p=act_proba)

                action_ind = int(action_ind.detach().cpu().numpy())

                action = self.action_space[action_ind]
                print("ACTION=", action)

                obs.append(rgb)
                actions.append(action_ind)
                
                # get best action in a confidence sense
                if mode=="train":
                    best_action = 4 # "DoNothing"
                    best_conf = conf_prev
                    for action_idx in [0,1,2,3]:
                        action_t = self.action_space[action_idx]
                        event_t = self.controller.step(action_t)
                        agent_position_t = np.array(list(event_t.metadata['agent']['position'].values())) + np.array([0.0, 0.675, 0.0])
                        turn_yaw_t, turn_pitch_t = self.utils.get_rotation_to_obj(obj_center, agent_position_t)
                        event_t = self.controller.step('TeleportFull', x=agent_position_t[0], y=agent_position_t[1], z=agent_position_t[2], rotation=dict(x=0.0, y=int(turn_yaw_t), z=0.0), horizon=int(turn_pitch_t))
                        rgb_t = event_t.frame
                        conf_t, _, _, _ = self.get_detectron_conf_center_obj(rgb_t, obj_mask, frame)
                        if conf_t is None:
                            conf_t = best_conf - 1 # dont want no detection
                        if conf_t > best_conf:
                            best_action = action_idx
                            best_conf = conf_t
                        _ = self.controller.step('TeleportFull', x=pos_s_prev[0], y=pos_s_prev[1], z=pos_s_prev[2], rotation=dict(x=0.0, y=int(turn_yaw_prev), z=0.0), horizon=int(turn_pitch_prev))

                    best_action = torch.LongTensor([best_action]).cuda()
                    total_loss += self.loss(act_proba, best_action)
                    num_obs += 1
                
                    print("BEST ACTION=", self.action_space[int(best_action.detach().cpu().numpy())])

                if not action=="DoNothing":
                    event = self.controller.step(action)
                    agent_position = np.array(list(event.metadata['agent']['position'].values())) + np.array([0.0, 0.675, 0.0])
                    turn_yaw, turn_pitch = self.utils.get_rotation_to_obj(obj_center, agent_position)
                    event = self.controller.step('TeleportFull', x=agent_position[0], y=agent_position[1], z=agent_position[2], rotation=dict(x=0.0, y=int(turn_yaw), z=0.0), horizon=int(turn_pitch))
                else:
                    print("Do nothing reached")
                    print("End confidence: ", conf_prev)
                    break

                if mode=="train":
                    pos_s_prev = agent_position
                    turn_yaw_prev = turn_yaw
                    turn_pitch_prev = turn_pitch

                rgb = event.frame
                conf_cur, obj_pred_classes, obj_mask_new, seg_im = self.get_detectron_conf_center_obj(rgb, obj_mask, frame)
                seg_ims.append(seg_im)
                if conf_cur is None:
                    conf_cur = conf_prev
                    seg_im = rgb
                else:
                    obj_mask = obj_mask_new
                
                if True:
                    plt.figure(1)
                    plt.clf()
                    plt.imshow(seg_im)
                    plt_name = self.homepath + f'/seg{frame}.png'
                    plt.savefig(plt_name)

                confs.append(conf_cur)

                conf_prev = conf_cur


                if conf_cur > self.conf_thresh_end:
                    print("CONFIDENCE THRESHOLD REACHED!")
                    print("End confidence: ", conf_cur)
                    break

                if frame >= self.max_frames - 1:
                    print("MAX FRAMES REACHED")
                    print("End confidence: ", conf_cur)
                    break

                frame += 1
                
                        
            return total_loss, obs, actions, seg_ims, confs

if __name__ == '__main__':
    # startx()
    Ai2Thor()


