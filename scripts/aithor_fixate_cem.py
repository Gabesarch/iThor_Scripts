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

import torch
import torch.nn as nn
import torch.nn.functional as F 
from argparse import Namespace

from nets.cemnet import CEMNet


print(os.getcwd())

def pci_records():
    records = []
    command = shlex.split('lspci -vmm')
    output = subprocess.check_output(command).decode()

    for devices in output.strip().split("\n\n"):
        record = {}
        records.append(record)
        for row in devices.split("\n"):
            key, value = row.split("\t")
            record[key.split(':')[0]] = value

    return records

def generate_xorg_conf(devices):
    xorg_conf = []

    device_section = """
Section "Device"
    Identifier     "Device{device_id}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "{bus_id}"
EndSection
"""
    server_layout_section = """
Section "ServerLayout"
    Identifier     "Layout0"
    {screen_records}
EndSection
"""
    screen_section = """
Section "Screen"
    Identifier     "Screen{screen_id}"
    Device         "Device{device_id}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual 1024 768
    EndSubSection
EndSection
"""
    screen_records = []
    for i, bus_id in enumerate(devices):
        xorg_conf.append(device_section.format(device_id=i, bus_id=bus_id))
        xorg_conf.append(screen_section.format(device_id=i, screen_id=i))
        screen_records.append('Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=i))
    
    xorg_conf.append(server_layout_section.format(screen_records="\n    ".join(screen_records)))

    output =  "\n".join(xorg_conf)
    return output

def _startx(display):
    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")

    devices = []
    for r in pci_records():
        if r.get('Vendor', '') == 'NVIDIA Corporation'\
                and r['Class'] in ['VGA compatible controller', '3D controller']:
            bus_id = 'PCI:' + ':'.join(map(lambda x: str(int(x, 16)), re.split(r'[:\.]', r['Slot'])))
            devices.append(bus_id)

    if not devices:
        raise Exception("no nvidia cards found")

    try:
        fd, path = tempfile.mkstemp()
        with open(path, "w") as f:
            f.write(generate_xorg_conf(devices))
        command = shlex.split("Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config %s :%s" % (path, display))
        proc = subprocess.Popen(command)
        atexit.register(lambda: proc.poll() is None and proc.kill())
        proc.wait()
    finally: 
        os.close(fd)
        os.unlink(path)

def startx(display=0):
    if 'DISPLAY' in os.environ:
        print("Skipping Xorg server - DISPLAY is already running at %s" % os.environ['DISPLAY'])
        return

    xthread = threading.Thread(target=_startx, args=(display,))
    xthread.daemon = True
    xthread.start()
    # wait for server to start
    time.sleep(4)



class Ai2Thor():
    def __init__(self):   
        self.visualize = False
        self.verbose = False
        self.save_imgs = True
        self.do_orbslam = False
        self.do_depth_noise = False
        self.makevideo = True
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

        np.random.seed(1)
        random.shuffle(mapnames)
        self.mapnames = mapnames
        self.num_episodes = len(self.mapnames)   

        self.max_iters = 100000
        self.max_frames = 10

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

        self.action_space = {0: "MoveLeft", 1: "MoveRight", 2: "MoveAhead", 3: "MoveBack"}
        self.num_actions = len(self.action_space)

        cfg_det = get_cfg()
        cfg_det.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_det.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
        cfg_det.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg_det.MODEL.DEVICE='cpu'
        self.cfg_det = cfg_det
        self.maskrcnn = DefaultPredictor(cfg_det)

        self.conf_thresh_detect = 0.7 # for initially detecting a low confident object
        self.conf_thresh_init = 0.8 # for after turning head toward object threshold

        self.small_classes = []
        self.rot_interval = 5.0
        self.radius_max = 3.5 #3 #1.75
        self.radius_min = 1.0 #1.25
        self.num_flat_views = 3
        self.num_any_views = 7
        self.num_views = 25

        self.obj_per_scene = 5

        # self.origin_quaternion = np.quaternion(1, 0, 0, 0)
        # self.origin_rot_vector = quaternion.as_rotation_vector(self.origin_quaternion) 

        self.homepath = f'/home/nel/gsarch/aithor/data/test'
        # self.basepath = '/home/nel/gsarch/replica_traj_bed'
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


        self.W = 256
        self.H = 256

        # self.fov = 90
        # hfov = float(self.fov) * np.pi / 180.
        # self.pix_T_camX = np.array([
        #     [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
        #     [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
        #     [0., 0.,  1, 0],
        #     [0., 0., 0, 1]])
        # self.pix_T_camX[0,2] = self.W/2.
        # self.pix_T_camX[1,2] = self.H/2.

        self.fov = 90
        self.camera_matrix = self.get_camera_matrix(self.W, self.H, self.fov)
        self.K = self.get_habitat_pix_T_camX(self.fov)

        self.init_network()

        self.run_episodes()
    
    def init_network(self):
        
        self.cemnet = CEMNet(h1=32, h2=64, fc_dim=1024, num_actions=self.num_actions)

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(params=self.cemnet.parameters(),lr=0.01)

    def batch_iteration(self,mapnames,NN,BATCH_SIZE):

        batch = {"rewards": [], "obs": [], "actions": []}
        episode_rewards = 0.0
        episode_steps = []

        iter_idx = 0
        while True:

            mapname = np.random.choice(self.mapnames)

            # self.basepath = self.homepath + f"/{mapname}_{episode}"
            # print("BASEPATH: ", self.basepath)

            # # self.basepath = f"/hdd/ayushj/habitat_data/{mapname}_{episode}"
            # if not os.path.exists(self.basepath):
            #     os.mkdir(self.basepath)

            self.controller = Controller(
                scene=mapname, 
                gridSize=0.25,
                width=self.W,
                height=self.H,
                fieldOfView= self.fov,
                renderObjectImage=True,
                renderDepthImage=True,
                )

            episode_rewards, obs, actions = self.run()

            print("Total reward for iter ",iter_idx," :",episode_rewards)

            self.controller.stop()
            time.sleep(1)

            if episode_rewards == False:
                continue

            batch["rewards"].append(episode_rewards)
            batch["obs"].append(obs)
            batch["actions"].append(actions) 

            iter_idx += 1   


            if len(batch) == BATCH_SIZE:

                yield batch
    
    def elite_batch(batch,percentile):

        rewards = batch["rewards"]
        obs = batch["obs"]
        actions = batch["actions"]

        rewards_mean = float(np.mean(rewards))

        rewards_boundary = np.percentile(rewards,percentile)

        rewards_mean = float(np.mean(rewards))

        training_obs = []
        training_actions = []

        for idx in rewards.shape[0]:
            reward_idx = rewards[idx]
            if reward_idx < rewards_boundary:
                continue

            training_obs.extend(obs[idx])
            training_actions.extend(actions[idx])

        obs_tensor = torch.FloatTensor(training_obs)
        act_tensor = torch.LongTensor(training_actions)

        return obs_tensor, act_tensor, rewards_mean, rewards_boundary

    def run_episodes(self):
        self.ep_idx = 0
        BATCH_SIZE=10
        percentile = 70
        # self.objects = []

        for iteration, batch in enumerate(self.batch_iteration(self.mapnames,self.cemnet,BATCH_SIZE)):
            print("ITERATION #", iteration)

            train, labels, mean_rewards, boudary = self.elite_batch(batch,percentile)

            optimizer.zero_grad()

            scores = self.cemnet(train, softmax=False)

            loss_value = self.loss(scores,labels)

            back_v = loss_value.backward()

            self.optimizer.step()

            print('rewards mean = ',mean_rewards)

            if iteration >= self.max_iters:
                print("MAX ITERS REACHED")
                break

            # if mean_rewards > 500:
            #     print('Accomplished!')
            #     break

    def save_datapoint(self, observations, data_path, viewnum, flat_view):
        if self.verbose:
            print("Print Sensor States.", self.agent.state.sensor_states)
        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        # st()
        depth = observations["depth_sensor"]
        agent_pos = observations["positions"] 
        agent_rot = observations["rotations"]
        # Assuming all sensors have same extrinsics
        color_sensor_pos = observations["positions"] 
        color_sensor_rot = observations["rotations"] 
        #print("POS ", agent_pos)
        #print("ROT ", color_sensor_rot)
        object_list = observations['object_list']

        # print(viewnum, agent_pos)
        # print(agent_rot)

        if False:
            plt.imshow(rgb)
            plt_name = f'/home/nel/gsarch/aithor/data/test/img_mask{viewnum}.png'
            plt.savefig(plt_name)
                
        save_data = {'flat_view': flat_view, 'objects_info': object_list,'rgb_camX':rgb, 'depth_camX': depth, 'sensor_pos': color_sensor_pos, 'sensor_rot': color_sensor_rot}
        
        with open(os.path.join(data_path, str(viewnum) + ".p"), 'wb') as f:
            pickle.dump(save_data, f)
        f.close()
    
    def quat_from_angle_axis(self, theta: float, axis: np.ndarray) -> np.quaternion:
        r"""Creates a quaternion from angle axis format

        :param theta: The angle to rotate about the axis by
        :param axis: The axis to rotate about
        :return: The quaternion
        """
        axis = axis.astype(np.float)
        axis /= np.linalg.norm(axis)
        return quaternion.from_rotation_vector(theta * axis)

    def safe_inverse_single(self,a):
        r, t = self.split_rt_single(a)
        t = np.reshape(t, (3,1))
        r_transpose = r.T
        inv = np.concatenate([r_transpose, -np.matmul(r_transpose, t)], 1)
        bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
        # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4) 
        inv = np.concatenate([inv, bottom_row], 0)
        return inv
    
    def split_rt_single(self,rt):
        r = rt[:3, :3]
        t = np.reshape(rt[:3, 3], 3)
        return r, t

    def eul2rotm(self, rx, ry, rz):
        # inputs are shaped B
        # this func is copied from matlab
        # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
        #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
        #        -sy            cy*sx             cy*cx]
        # rx = np.expand_dims(rx, axis=1)
        # ry = np.expand_dims(ry, axis=1)
        # rz = np.expand_dims(rz, axis=1)
        # st()
        # these are B x 1
        sinz = np.sin(rz)
        siny = np.sin(ry)
        sinx = np.sin(rx)
        cosz = np.cos(rz)
        cosy = np.cos(ry)
        cosx = np.cos(rx)
        r11 = cosy*cosz
        r12 = sinx*siny*cosz - cosx*sinz
        r13 = cosx*siny*cosz + sinx*sinz
        r21 = cosy*sinz
        r22 = sinx*siny*sinz + cosx*cosz
        r23 = cosx*siny*sinz - sinx*cosz
        r31 = -siny
        r32 = sinx*cosy
        r33 = cosx*cosy

        r = np.array([[r11, r12, r13], 
                    [r21, r22, r23],
                    [r31, r32, r33]
                    ])
        # r1 = np.stack([r11,r12,r13],axis=2)
        # r2 = np.stack([r21,r22,r23],axis=2)
        # r3 = np.stack([r31,r32,r33],axis=2)
        # r = np.concatenate([r1,r2,r3],axis=1)
        return r

    def rotm2eul(self,r):
        # r is Bx3x3, or Bx4x4
        r00 = r[0,0]
        r10 = r[1,0]
        r11 = r[1,1]
        r12 = r[1,2]
        r20 = r[2,0]
        r21 = r[2,1]
        r22 = r[2,2]
        
        ## python guide:
        # if sy > 1e-6: # singular
        #     x = math.atan2(R[2,1] , R[2,2])
        #     y = math.atan2(-R[2,0], sy)
        #     z = math.atan2(R[1,0], R[0,0])
        # else:
        #     x = math.atan2(-R[1,2], R[1,1])
        #     y = math.atan2(-R[2,0], sy)
        #     z = 0
        
        sy = np.sqrt(r00*r00 + r10*r10)
        
        cond = (sy > 1e-6)
        rx = np.where(cond, np.arctan2(r21, r22), np.arctan2(-r12, r11))
        ry = np.where(cond, np.arctan2(-r20, sy), np.arctan2(-r20, sy))
        rz = np.where(cond, np.arctan2(r10, r00), np.zeros_like(r20))

        # rx = torch.atan2(r21, r22)
        # ry = torch.atan2(-r20, sy)
        # rz = torch.atan2(r10, r00)
        # rx[cond] = torch.atan2(-r12, r11)
        # ry[cond] = torch.atan2(-r20, sy)
        # rz[cond] = 0.0
        return rx, ry, rz

    def get_habitat_pix_T_camX(self, fov):
        hfov = float(self.fov) * np.pi / 180.
        pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        return pix_T_camX

    def get_camera_matrix(self, width, height, fov):
        """Returns a camera matrix from image size and fov."""
        xc = (width - 1.) / 2.
        zc = (height - 1.) / 2.
        f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
        camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
        camera_matrix = Namespace(**camera_matrix)
        return camera_matrix
    
    def run2(self):
        event = self.controller.step('GetReachablePositions')
        for obj in event.metadata['objects']:
            if obj['objectType'] not in self.objects:
                self.objects.append(obj['objectType'])

    def get_detectron_conf_center_obj(self,im):
        im = Image.fromarray(im, mode="RGB")
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

        # plt.imshow(im)
        # plt.show()

        outputs = self.maskrcnn(im)

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg_det.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs['instances'].to("cpu"))
        seg_im = out.get_image()

        if True:
            plt.imshow(seg_im)
            plt_name = '/home/nel/gsarch/aithor/data/test/seg.png'
            plt.savefig(plt_name)
            # plt.show()

        pred_masks = outputs['instances'].pred_masks
        pred_scores = outputs['instances'].scores
        pred_classes = outputs['instances'].pred_classes

        W2_low = self.W//2 - 5
        W2_high = self.W//2 + 5
        H2_low = self.H//2 - 5
        H2_high = self.H//2 + 5

        ind_obj = None
        max_overlap = 0
        for idx in range(pred_masks.shape[0]):
            pred_mask_cur = pred_masks[idx]
            pred_masks_center = pred_mask_cur[W2_low:W2_high, H2_low:H2_high]
            print(torch.sum(pred_masks_center))
            if torch.sum(pred_masks_center) > max_overlap:
                ind_obj = idx
                max_overlap = torch.sum(pred_masks_center)
        if ind_obj is None:
            return None

        print("OBJ CLASS ID=", int(pred_classes[ind_obj].detach().cpu().numpy()))
        # pred_boxes = outputs['instances'].pred_boxes.tensor
        # pred_classes = outputs['instances'].pred_classes
        # pred_scores = outputs['instances'].scores
        obj_score = pred_scores[ind_obj]

        return obj_score

    def detect_object_centroid(self, im, event):

        im = Image.fromarray(im, mode="RGB")
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

        # plt.imshow(im)
        # plt.show()

        outputs = self.maskrcnn(im)

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg_det.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs['instances'].to("cpu"))
        seg_im = out.get_image()

        if True:
            plt.imshow(seg_im)
            plt_name = '/home/nel/gsarch/aithor/data/test/seg_init.png'
            plt.savefig(plt_name)
            # plt.show()

        pred_masks = outputs['instances'].pred_masks
        pred_boxes = outputs['instances'].pred_boxes.tensor
        pred_classes = outputs['instances'].pred_classes
        pred_scores = outputs['instances'].scores

        # obj_ids = []
        obj_catids = []
        obj_scores = []
        obj_masks = []
        # obj_all_catids = []
        # obj_all_scores = []
        # obj_all_boxes = []
        for segs in range(len(pred_masks)):
            if pred_scores[segs] <= self.conf_thresh_detect:
                # obj_ids.append(segs)
                obj_catids.append(pred_classes[segs].item())
                obj_scores.append(pred_scores[segs].item())
                obj_masks.append(pred_masks[segs])

                # obj_all_catids.append(pred_classes[segs].item())
                # obj_all_scores.append(pred_scores[segs].item())
                # y, x = torch.where(pred_masks[segs])
                # pred_box = torch.Tensor([min(y), min(x), max(y), max(x)]) # ymin, xmin, ymax, xmaxs
                # obj_all_boxes.append(pred_box)

        # print("MASKS ", len(pred_masks))
        # print("VALID ", len(obj_scores))
        # print(obj_scores)
        # print(pred_scores.shape)

        eulers_xyz_rad = np.radians(np.array([event.metadata['agent']['cameraHorizon'], event.metadata['agent']['rotation']['y'], 0.0]))

        rx = eulers_xyz_rad[0]
        ry = eulers_xyz_rad[1]
        rz = eulers_xyz_rad[2]
        rotation_ = self.eul2rotm(-rx, -ry, rz)

        translation_ = np.array(list(event.metadata['agent']['position'].values())) + np.array([0.0, 0.675, 0.0])
        # need to invert since z is positive here by convention
        translation_[2] =  -translation_[2]

        T_world_cam = np.eye(4)
        T_world_cam[0:3,0:3] =  rotation_
        T_world_cam[0:3,3] = translation_

        if not obj_masks:
            return None
        else: 

            # want an object not on the edges of the image
            sum_interior = 0
            while sum_interior==0:
                if len(obj_masks)==0:
                    return None
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

            print("MIDPOINT=", xyz_obj_mid)

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
            #     if obj_category_name not in self.include_classes: # or not obj_meta['visible']:
            #         continue

            #     obj_instance_mask = instance_masks[object_id]
            #     obj_instance_detection2D = instance_detections2D[object_id] # [start_x, start_y, end_x, end_y]
            #     obj_instance_detection2D = np.array([obj_instance_detection2D[1], obj_instance_detection2D[0], obj_instance_detection2D[3], obj_instance_detection2D[2]])  # ymin, xmin, ymax, xmax

            #     if True:
            #         print(object_id)
            #         print(np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values())))
            #         # plt.imshow(obj_instance_mask)
            #         # plt_name = f'/home/nel/gsarch/aithor/data/test/img_mask{s}.png'
            #         # plt.savefig(plt_name)

            #     obj_center_axisAligned = np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))
            #     obj_center_axisAligned[2] = -obj_center_axisAligned[2]
            #     obj_size_axisAligned = np.array(list(obj_meta['axisAlignedBoundingBox']['size'].values()))                        

            return xyz_obj_mid

    def get_rotation_to_obj(self, obj_center, pos_s):
        # YAW calculation - rotate to object
        agent_to_obj = np.squeeze(obj_center) - pos_s 
        agent_local_forward = np.array([0, 0, 1.0]) 
        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
        flat_to_obj /= flat_dist_to_obj

        det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

        turn_yaw = np.degrees(turn_angle)

        turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))

        return turn_yaw, turn_pitch

    def run(self):
        
        event = self.controller.step('GetReachablePositions')
        if not event.metadata['reachablePositions']:
            # Different versions this is empty/full
            event = self.controller.step(action='MoveAhead')
        self.nav_pts = event.metadata['reachablePositions']
        self.nav_pts = np.array([list(d.values()) for d in self.nav_pts])
        np.random.seed(1)
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
        while True: #successes < self.obj_per_scene and meta_obj_idx <= len(event.metadata['objects']) - 1: 
            if meta_obj_idx > len(event.metadata['objects']) - 1:
                print("OUT OF OBJECT... RETURNING")
                return False, False, False
                
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

            turn_yaw, turn_pitch = self.get_rotation_to_obj(obj_center, pos_s)
            event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1], z=pos_s[2], rotation=dict(x=0.0, y=int(turn_yaw), z=0.0), horizon=int(turn_pitch))
            rgb = event.frame

            # get object center of a low confidence object
            obj_center = self.detect_object_centroid(rgb, event)

            if obj_center is None:
                print("NO LOW CONFIDENCE OBJECTS... SKIPPING...")
                continue

            # initialize object in center of FOV
            turn_yaw, turn_pitch = self.get_rotation_to_obj(obj_center, pos_s)
            event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1], z=pos_s[2], rotation=dict(x=0.0, y=int(turn_yaw), z=0.0), horizon=int(turn_pitch))
            rgb = event.frame
            init_conf = self.get_detectron_conf_center_obj(rgb)
            conf_prev = init_conf
            if conf > self.conf_thresh_init:
                print("HIGH INITIAL CONFIDENCE... SKIPPING...")
                continue
            
            obs = []
            actions = []
            episode_rewards = 0.0
            frame = 0
            while True:

                rgb_tensor = torch.FloatTensor([rgb])
                
                actions_probability = self.cemnet(rgb_tensor, softmax=True)

                act_proba = actions_probability.data.numpy()[0]

                action_ind = np.random.choice(len(act_proba),p=act_proba)

                action = self.action_space[action_indxs]

                event = self.controller.step(action)
                rgb = event.frame

                conf_cur = self.get_detectron_conf_center_obj(rgb)

                reward = (conf_cur - conf_prev)/(1 - init_conf) # normalize by intial confidence to account for differences in starting confidence
                conf_prev = conf_cur
                
                episode_rewards += reward

                obs.append(rgb)
                actions.append(action_ind)

                if conf_cur > self.conf_thresh_end:
                    print("CONFIDENCE THRESHOLD REACHED!")
                    print("End confidence: ", conf_cur)
                    break

                if frame >= self.max_frames - 1:
                    print("MAX FRAMES REACHED")
                    print("End confidence: ", conf_cur)
                    break

                frame += 1

            return episode_rewards, obs, actions

            #     eulers_xyz_rad = np.radians(np.array([event.metadata['agent']['cameraHorizon'], event.metadata['agent']['rotation']['y'], 0.0]))

            #     rx = eulers_xyz_rad[0]
            #     ry = eulers_xyz_rad[1]
            #     rz = eulers_xyz_rad[2]
            #     rotation_r_matrix = self.eul2rotm(-rx, -ry, rz)

            #     agent_position = np.array(list(event.metadata['agent']['position'].values())) + np.array([0.0, 0.675, 0.0])
            #     # need to invert since z is positive here by convention
            #     agent_position[2] =  -agent_position[2]


            #     observations["positions"] = agent_position

            #     observations["rotations"] = rotation_r_matrix

            #     # rt_4x4 = np.eye(4)
            #     # rt_4x4[0:3,0:3] = observations["rotations"]
            #     # rt_4x4[0:3,3] = observations["positions"]
            #     # rt_4x4_inv = self.safe_inverse_single(rt_4x4)
            #     # r, t = self.split_rt_single(rt_4x4_inv)

            #     # observations["positions"] = r

            #     # observations["positions"] = t

            #     # observations["rotations_euler"] = np.array([rx, ry, rz]) #rotation_r.as_euler('xyz', degrees=True)

            #     observations["color_sensor"] = rgb
            #     observations["depth_sensor"] = event.depth_frame
            #     observations["semantic_sensor"] = event.instance_segmentation_frame

            #     if True:
            #         im = Image.fromarray(rgb, mode="RGB")
            #         im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            #         outputs = self.get_detectron_out(rgb)
            #         v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg_det.DATASETS.TRAIN[0]), scale=1.2)
            #         out = v.draw_instance_predictions(outputs['instances'].to("cpu"))
            #         seg_im = out.get_image()
            #         plt.imshow(seg_im)
            #         plt_name = f'/home/nel/gsarch/aithor/data/test/img_true{s}{b}.png'
            #         plt.savefig(plt_name)

            #     # print("Processed image #", cnt, " for object ", obj['objectType'])

            #     semantic = event.instance_segmentation_frame
            #     object_id_to_color = event.object_id_to_color
            #     color_to_object_id = event.color_to_object_id

            #     obj_ids = np.unique(semantic.reshape(-1, semantic.shape[2]), axis=0)

            #     instance_masks = event.instance_masks
            #     instance_detections2D = event.instance_detections2D
                
            #     obj_metadata_IDs = []
            #     for obj_m in event.metadata['objects']: #objects:
            #         obj_metadata_IDs.append(obj_m['objectId'])
                
            #     object_list = []
            #     for obj_idx in range(obj_ids.shape[0]):
            #         try:
            #             obj_color = tuple(obj_ids[obj_idx])
            #             object_id = color_to_object_id[obj_color]
            #         except:
            #             # print("Skipping ", object_id)
            #             continue

            #         if object_id not in obj_metadata_IDs:
            #             # print("Skipping ", object_id)
            #             continue

            #         obj_meta_index = obj_metadata_IDs.index(object_id)
            #         obj_meta = event.metadata['objects'][obj_meta_index]
            #         obj_category_name = obj_meta['objectType']
                    
            #         # continue if not visible or not in include classes
            #         if obj_category_name not in self.include_classes or not obj_meta['visible']:
            #             continue

            #         obj_instance_mask = instance_masks[object_id]
            #         obj_instance_detection2D = instance_detections2D[object_id] # [start_x, start_y, end_x, end_y]
            #         obj_instance_detection2D = np.array([obj_instance_detection2D[1], obj_instance_detection2D[0], obj_instance_detection2D[3], obj_instance_detection2D[2]])  # ymin, xmin, ymax, xmax

            #         if False:
            #             print(object_id)
            #             plt.imshow(obj_instance_mask)
            #             plt_name = f'/home/nel/gsarch/aithor/data/test/img_mask{s}.png'
            #             plt.savefig(plt_name)

            #         obj_center_axisAligned = np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))
            #         obj_center_axisAligned[2] = -obj_center_axisAligned[2]
            #         obj_size_axisAligned = np.array(list(obj_meta['axisAlignedBoundingBox']['size'].values()))
                    
            #         # print(obj_category_name)

            #         if self.verbose: 
            #             print("Saved class name is : ", obj_category_name)

            #         obj_data = {'instance_id': object_id, 'category_id': object_id, 'category_name': obj_category_name,
            #                         'bbox_center': obj_center_axisAligned, 'bbox_size': obj_size_axisAligned,
            #                             'mask_2d': obj_instance_mask, 'box_2d': obj_instance_detection2D}
            #         # object_list.append(obj_instance)
            #         object_list.append(obj_data)
                
            #     observations["object_list"] = object_list

            #     # check if object visible (make sure agent is not behind a wall)
            #     obj_id = obj['objectId']
            #     obj_id_to_color = object_id_to_color[obj_id]
            #     # if np.sum(obj_ids==object_id_to_color[obj_id]) > 0:
            #     if self.verbose:
            #         print("episode is valid......")
            #     episodes.append(observations)

            # st()  
            # if len(episodes) >= self.num_views:
            #     print(f'num episodes: {len(episodes)}')
            #     data_folder = obj['name']
            #     data_path = os.path.join(self.basepath, data_folder)
            #     print("Saving to ", data_path)
            #     os.mkdir(data_path)
            #     # flat_obs = np.random.choice(episodes, self.num_views, replace=False)
            #     np.random.seed(1)
            #     rand_inds = np.sort(np.random.choice(len(episodes), self.num_views, replace=False))
            #     bool_inds = np.zeros(len(episodes), dtype=bool)
            #     bool_inds[rand_inds] = True
            #     flat_obs = np.array(episodes)[bool_inds]
            #     flat_obs = list(flat_obs)
            #     viewnum = 0
            #     if False:
            #         self.generate_xyz_habitatCamXs(flat_obs)
            #     for obs in flat_obs:
            #         self.save_datapoint(obs, data_path, viewnum, True)
            #         viewnum += 1
            #     print("SUCCESS #", successes)
            #     successes += 1
            # else:
            #     print("Not enough episodes:", len(episodes))

if __name__ == '__main__':
    # startx()
    Ai2Thor()


