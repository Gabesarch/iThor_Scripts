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

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F 
from argparse import Namespace

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
        self.num_episodes = 3 #len(self.mapnames)   

        self.ignore_classes = []  
        # classes to save   
        # self.include_classes = [
        #     'ShowerDoor', 'Cabinet', 'CounterTop', 'Sink', 'Towel', 'HandTowel', 'TowelHolder', 'SoapBar', 
        #     'ToiletPaper', 'ToiletPaperHanger', 'HandTowelHolder', 'SoapBottle', 'GarbageCan', 'Candle', 'ScrubBrush', 
        #     'Plunger', 'SinkBasin', 'Cloth', 'SprayBottle', 'Toilet', 'Faucet', 'ShowerHead', 'Box', 'Bed', 'Book', 
        #     'DeskLamp', 'BasketBall', 'Pen', 'Pillow', 'Pencil', 'CellPhone', 'KeyChain', 'Painting', 'CreditCard', 
        #     'AlarmClock', 'CD', 'Laptop', 'Drawer', 'SideTable', 'Chair', 'Blinds', 'Desk', 'Curtains', 'Dresser', 
        #     'Watch', 'Television', 'WateringCan', 'Newspaper', 'FloorLamp', 'RemoteControl', 'HousePlant', 'Statue', 
        #     'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'BaseballBat', 'TennisRacket', 'VacuumCleaner', 'Mug', 'ShelvingUnit', 
        #     'Shelf', 'StoveBurner', 'Apple', 'Lettuce', 'Bottle', 'Egg', 'Microwave', 'CoffeeMachine', 'Fork', 'Fridge', 
        #     'WineBottle', 'Spatula', 'Bread', 'Tomato', 'Pan', 'Cup', 'Pot', 'SaltShaker', 'Potato', 'PepperShaker', 
        #     'ButterKnife', 'StoveKnob', 'Toaster', 'DishSponge', 'Spoon', 'Plate', 'Knife', 'DiningTable', 'Bowl', 
        #     'LaundryHamper', 'Vase', 'Stool', 'CoffeeTable', 'Poster', 'Bathtub', 'TissueBox', 'Footstool', 'BathtubBasin', 
        #     'ShowerCurtain', 'TVStand', 'Boots', 'RoomDecor', 'PaperTowelRoll', 'Ladle', 'Kettle', 'Safe', 'GarbageBag', 'TeddyBear', 
        #     'TableTopDecor', 'Dumbbell', 'Desktop', 'AluminumFoil', 'Window']

        # These are all classes shared between aithor and coco
        self.include_classes = [
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
            'TeddyBear', 
            ]


        self.maskrcnn_to_ithor = {
            81:'Sink', 
            70:'Toilet', 65:'Bed', 84:'Book', 
            77:'CellPhone', 
            85:'AlarmClock', 73:'Laptop', 62:'Chair',
            72:'Television', 75:'RemoteControl', 64:'HousePlant', 
            62:'Ottoman', 62:'ArmChair', 63:'Sofa', 39:'BaseballBat', 43:'TennisRacket', 47:'Mug', 
            48:'Apple', 40:'Bottle', 69:'Microwave', 43:'Fork', 72:'Fridge', 
            44:'WineBottle', 47:'Cup', 
            49:'ButterKnife', 80:'Toaster', 50:'Spoon', 49:'Knife', 67:'DiningTable', 51:'Bowl', 
            86:'Vase', 
            88:'TeddyBear', 
        }

        self.ithor_to_maskrcnn = {
            'Sink':81, 
            'Toilet':70, 'Bed':65, 'Book':84, 
            'CellPhone':77, 
            'AlarmClock':85, 'Laptop':73, 'Chair':62,
            'Television':72, 'RemoteControl':75, 'HousePlant':64, 
            'Ottoman':62, 'ArmChair':62, 'Sofa':63, 'BaseballBat':39, 'TennisRacket':43, 'Mug':47, 
            'Apple':53, 'Bottle':44, 'Microwave':78, 'Fork':48, 'Fridge':82, 
            'WineBottle':44, 'Cup':47, 
            'ButterKnife':49, 'Toaster':80, 'Spoon':50, 'Knife':49, 'DiningTable':67, 'Bowl':51, 
            'Vase':86, 
            'TeddyBear':88, 
        }

        self.maskrcnn_to_catname = {
            81:'sink', 
            67:'dining table', 65:'bed', 84:'book', 
            77:'cell phone', 70: 'toilet',
            85:'clock', 73:'laptop', 62:'chair',
            72:'tv', 75:'remote', 64:'potted plant', 
            63:'couch', 39:'baseball bat', 43:'tennis racket', 47:'cup', 
            53:'apple', 44:'bottle', 78:'microwave', 48:'fork', 82:'refrigerator', 
            46:'wine glass', 
            49:'knife', 79:'oven', 80:'toaster', 50:'spoon', 67:'dining table', 51:'bowl', 
            86:'vase', 
            88:'teddy bear', 
        }

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

        cfg_det = get_cfg()
        cfg_det.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_det.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
        cfg_det.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg_det.MODEL.DEVICE='cpu'
        self.cfg_det = cfg_det
        self.maskrcnn = DefaultPredictor(cfg_det)

        self.maskrcnn_to_detectron = MetadataCatalog.get(self.cfg_det.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id

        self.small_classes = []
        self.rot_interval = 5.0
        self.radius_max = 1.1 #3 #1.75
        self.radius_min = 1 #1.25
        self.num_flat_views = 3
        self.num_any_views = 7
        self.num_views = 25

        self.obj_per_scene = 5

        # self.origin_quaternion = np.quaternion(1, 0, 0, 0)
        # self.origin_rot_vector = quaternion.as_rotation_vector(self.origin_quaternion) 

        # self.homepath = f'/home/sirdome/katefgroup/gsarch/ithor/data/test'
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

        self.fov = 90
        hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2.
        self.pix_T_camX[1,2] = self.H/2.

        self.controller = Controller(
            scene='FloorPlan30', # will change 
            gridSize=0.25,
            width=self.W,
            height=self.H,
            fieldOfView= self.fov,
            renderObjectImage=True,
            renderDepthImage=True,
            )


        self.run_episodes()

    def run_episodes(self):
        self.ep_idx = 0
        # self.objects = []
        
        for episode in range(self.num_episodes):
            print("STARTING EPISODE ", episode)

            mapname = self.mapnames[episode]
            print("MAPNAME=", mapname)

            self.controller.reset(scene=mapname)
            # self.controller.start()
            
            self.basepath = self.homepath + f"/{mapname}_{episode}"
            print("BASEPATH: ", self.basepath)

            # self.basepath = f"/hdd/ayushj/habitat_data/{mapname}_{episode}"
            if not os.path.exists(self.basepath):
                os.mkdir(self.basepath)

            self.run()

            self.ep_idx += 1

        self.controller.stop()
        time.sleep(1)

        dict_keys = list(self.obj_conf_dict.keys())
        for obj_idx in range(len(self.obj_conf_dict)):
            key = dict_keys[obj_idx]
            obj_cur_array = np.array(self.obj_conf_dict[key])
            obj_cur_array_onlygood = obj_cur_array[obj_cur_array[:,0]==1][:,1]
            plt.figure(1)
            plt.clf()
            bins = np.arange(-180, 180, 10)
            plt.hist(obj_cur_array_onlygood)
            plt.title(key)
            plt.xlabel('Degrees to canonical pose')
            plt.ylabel('Number of times prediction is correct')
            plt_name = self.homepath + f'/hist_{key}.png'
            plt.savefig(plt_name)

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
    
    def generate_xyz_habitatCamXs(self, flat_obs):

        pix_T_camX = self.pix_T_camX
        xyz_camXs = []
        for i in range(2): #depth_camXs.shape[0]):
            K = pix_T_camX
            xs, ys = np.meshgrid(np.linspace(-1*self.W/2.,1*self.W/2.,self.W), np.linspace(1*self.W/2.,-1*self.W/2.,self.W))
            depth = flat_obs[i]['depth_sensor'].reshape(1,self.W,self.W)
            if i>0:
                rotation_X = flat_obs[i]['rotations'] #quaternion.as_rotation_matrix(rot)
                pos = flat_obs[i]['positions']
                euler_rot_X = flat_obs[i]["rotations_euler"]
                # euler_rot_X_rad = np.radians(flat_obs[i]["rotations_euler"])
                origin_T_camX_4x4 = np.eye(4)
                origin_T_camX_4x4[0:3,0:3] = rotation_X
                origin_T_camX_4x4[0:3,3] = pos
                origin_T_camX = rotation_X
                camX_T_origin_4x4 = self.safe_inverse_single(origin_T_camX_4x4)
                camX0_T_camX = np.matmul(camX0_T_origin, origin_T_camX)
                camX0_T_camX_4x4 = np.matmul(camX0_T_origin_4x4, origin_T_camX_4x4)
                # camX0_T_camX = np.matmul(rotation_0, np.linalg.inv(origin_T_camX))
                # camX0_T_camX_4x4 = np.matmul(origin_T_camX0_4x4, camX_T_origin_4x4)
                # r = R.from_matrix(camX0_T_camX)
                # print("CamX0_T_camX CHECK: ", r.as_euler('xyz', degrees=True))
                # r = R.from_matrix(origin_T_camX_4x4[0:3,0:3])
                # print("CamX0_T_camX CHECK: ", r.as_euler('xyz', degrees=True))
                # r = R.from_matrix(rotation_X)
                # rx = euler_rot_X_rad[0]
                # ry = euler_rot_X_rad[1]
                # rz = euler_rot_X_rad[2]
                # rotm = self.eul2rotm(rx, ry, rz)
                rx, ry, rz = self.rotm2eul(rotation_X)
                print("EULER ACTUAL: ", euler_rot_X)
                print("EULER OBTAINED: ", rx, ry, rz)
                
                rx, ry, rz = self.rotm2eul(camX0_T_camX)
                # print("origin_T_camx CHECK: ", r.as_euler('xyz', degrees=True))s
                print("EULER SUBTRACT: ", euler_rot_0 - euler_rot_X)
                print("EULER OBTAINED: ", rx, ry, rz)
                # st()
            elif i==0:
                rotation_0 = flat_obs[i]['rotations'] #quaternion.as_rotation_matrix(rot)
                pos = flat_obs[i]['positions']
                euler_rot_0 = flat_obs[i]["rotations_euler"]
                origin_T_camX0_4x4 = np.eye(4)
                origin_T_camX0_4x4[0:3,0:3] = rotation_0
                origin_T_camX0_4x4[0:3,3] = pos
                camX0_T_origin_4x4 = self.safe_inverse_single(origin_T_camX0_4x4)
                camX0_T_origin = np.linalg.inv(rotation_0)

            xs = xs.reshape(1,self.W,self.W)
            ys = ys.reshape(1,self.W,self.W)

            xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
            xys = xys.reshape(4, -1)
            xy_c0 = np.matmul(np.linalg.inv(K), xys)
            xyz_camX = xy_c0.T[:,:3]
            xyz_camXs.append(xyz_camX)
            if i ==1:
                xyz = np.expand_dims(xyz_camX, axis=0)
                B, N, _ = list(xyz.shape)
                ones = np.ones_like(xyz[:,:,0:1])
                xyz1 = np.concatenate([xyz, ones], 2)
                xyz1_t = np.transpose(xyz1, (0, 2, 1))
                # this is B x 4 x N
                xyz2_t = np.matmul(camX0_T_camX_4x4, xyz1_t)
                xyz2 = np.transpose(xyz2_t, (0, 2, 1))
                xyz2 = np.squeeze(xyz2[:,:,:3])
                # xyz2 = xyz_camX

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        xs = xyz2[:,0]
        ys = xyz2[:,1]
        zs = xyz2[:,2]
        # ax.scatter(xs, ys, zs)
        plt.plot(xs, zs, 'x', color = 'red')
        xyz3 = xyz_camXs[0]
        xs = xyz3[:,0]
        ys = xyz3[:,1]
        zs = xyz3[:,2]
        # ax.scatter(xs, ys, zs)
        plt.plot(xs, zs, 'o', color = 'blue')
        plt_name = '/home/nel/gsarch/aithor/data/pointcloud.png'
        plt.savefig(plt_name)
        st()



        return np.stack(xyz_camXs)

    def get_detectron_conf_center_obj(self,im, obj_mask, frame=None):
        im = Image.fromarray(im, mode="RGB")
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

        # plt.imshow(im)
        # plt.show()

        outputs = self.maskrcnn(im)

        pred_masks = outputs['instances'].pred_masks
        pred_scores = outputs['instances'].scores
        pred_classes = outputs['instances'].pred_classes

        len_pad = 5

        W2_low = self.W//2 - len_pad
        W2_high = self.W//2 + len_pad
        H2_low = self.H//2 - len_pad
        H2_high = self.H//2 + len_pad

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg_det.DATASETS.TRAIN[0]), scale=1.0)
        out = v.draw_instance_predictions(outputs['instances'].to("cpu"))
        seg_im = out.get_image()

        if False:
            plt.figure(1)
            plt.clf()
            plt.imshow(seg_im)
            plt_name = self.homepath + f'/seg_all{frame}.png'
            plt.savefig(plt_name)

        if False:
            seg_im[W2_low:W2_high, H2_low:H2_high,:] = 0.0
            plt.figure(1)
            plt.clf()
            plt.imshow(seg_im)
            plt_name = self.homepath + f'/seg_all_mask{frame}.png'
            plt.savefig(plt_name)

        ind_obj = None
        # max_overlap = 0
        sum_obj_mask = np.sum(obj_mask)
        mask_sum_thresh = 2000
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
            return None, None, None

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

        return obj_score, obj_pred_classes, seg_im
    
    def run(self):
        event = self.controller.step('GetReachablePositions')
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
        for obj in objects: #event.metadata['objects']: #objects:
            obj = objects[objects_inds[meta_obj_idx]]
            meta_obj_idx += 1
            print("Object is ", obj['objectType'])
            # if obj['name'] in ['Microwave_b200e0bc']:
            #     print(obj['name'])
            # else:
            #     continue
            # print(obj['name'])

            if obj['objectType'] not in self.include_classes:
                print("Continuing... Invalid Object")
                continue

            if not obj['pickupable']:
                print("Object ", obj['objectType'], " has no object aligned BB")
                continue

            obj_id = obj['objectId']
            obj_class = obj['objectType']
            maskrcnn_obj_id = self.ithor_to_maskrcnn[obj_class]
            maskrcnn_obj_catname = self.maskrcnn_to_catname[maskrcnn_obj_id]
            maskrcnn_obj_id = self.maskrcnn_to_detectron[maskrcnn_obj_id]
            # Calculate distance to object center
            obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))

            obj_oriented_bb = np.squeeze(np.array(list(obj['objectOrientedBoundingBox'].values())))

            ax_oriented_bb = np.array(obj['axisAlignedBoundingBox']['cornerPoints'])

            if False:
                plt.figure(2)
                plt.clf()
                plt.plot(obj_oriented_bb[:,2], obj_oriented_bb[:,0], 'x', color='red')
                plt.plot(ax_oriented_bb[:,2], ax_oriented_bb[:,0], 'x', color='blue')
                plt.plot(obj_oriented_bb[0,2], obj_oriented_bb[0,0], 'x', color='green')
                plt.plot(obj_oriented_bb[1,2], obj_oriented_bb[1,0], 'x', color='green')
                plt_name = self.homepath + '/bb.png'
                plt.savefig(plt_name)


            len_x = obj_oriented_bb[0,0] - obj_oriented_bb[1,0]
            len_z = obj_oriented_bb[0,2] - obj_oriented_bb[1,2]
            angle = np.degrees(np.tan(len_z/len_x))
                        
            obj_center = np.expand_dims(obj_center, axis=0)
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

            nbins = 50
            bins = np.linspace(-180, 180, nbins+1)
            bin_yaw = np.digitize(valid_yaw, bins)

            num_valid_bins = np.unique(bin_yaw).size            

            if False:
                import matplotlib.cm as cm
                colors = iter(cm.rainbow(np.linspace(0, 1, nbins)))
                plt.figure(2)
                plt.clf()
                print(np.unique(bin_yaw))
                plt.plot(obj_oriented_bb[:,2], obj_oriented_bb[:,0], 'x', color='red')
                plt.plot(ax_oriented_bb[:,2], ax_oriented_bb[:,0], 'x', color='blue')
                plt.plot(obj_oriented_bb[0,2], obj_oriented_bb[0,0], 'x', color='green')
                plt.plot(obj_oriented_bb[1,2], obj_oriented_bb[1,0], 'x', color='green')
                for bi in range(nbins):
                    cur_bi = np.where(bin_yaw==(bi+1))
                    points = valid_pts[cur_bi]
                    x_sample = points[:,0]
                    z_sample = points[:,2]
                    if not list(x_sample):
                        continue
                    plt.plot(z_sample, x_sample, 'o', color = next(colors))
                    plt_name = plt_name = self.homepath + '/valid.png'
                    plt.savefig(plt_name)
                    print(valid_yaw[cur_bi])
                    print("ANGLE=", angle)
                    print("travel angle=", valid_yaw[cur_bi] + angle)
                    print("travel angle2=", valid_yaw[cur_bi] - angle)
                    break
                st()
                plt.plot(self.nav_pts[:,2], self.nav_pts[:,0], 'x', color='red')
                plt.plot(obj_center[:,2], obj_center[:,0], 'x', color = 'black')
                

            if num_valid_bins == 0:
                continue

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

                    event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1], z=pos_s[2], rotation=dict(x=0.0, y=int(turn_yaw), z=0.0), horizon=int(turn_pitch))

                    rgb = event.frame

                    instance_masks = event.instance_masks
                    obj_mask = instance_masks[obj_id]

                    obj_viewing_orientation = valid_yaw[s_ind] - angle
                    print("viewing orientation=", obj_viewing_orientation)

                    conf, pred_class, seg_im = self.get_detectron_conf_center_obj(rgb, obj_mask, frame=cnt)

                    if conf is None:
                        conf = 0
                        seg_im = rgb
                        self.obj_conf_dict[maskrcnn_obj_catname].append([0, obj_viewing_orientation])
                    else:
                        # maskrcnn_obj_id = self.ithor_to_maskrcnn(obj_class)
                        # maskrcnn_obj_catname = self.maskrcnn_to_catname(maskrcnn_obj_id)
                        if maskrcnn_obj_id == pred_class:
                            self.obj_conf_dict[maskrcnn_obj_catname].append([1, obj_viewing_orientation])
                        else:
                            self.obj_conf_dict[maskrcnn_obj_catname].append([0, obj_viewing_orientation])

                    eulers_xyz_rad = np.radians(np.array([event.metadata['agent']['cameraHorizon'], event.metadata['agent']['rotation']['y'], 0.0]))

                    rx = eulers_xyz_rad[0]
                    ry = eulers_xyz_rad[1]
                    rz = eulers_xyz_rad[2]
                    rotation_r_matrix = self.eul2rotm(-rx, -ry, rz)

                    agent_position = np.array(list(event.metadata['agent']['position'].values())) + np.array([0.0, 0.675, 0.0])
                    # need to invert since z is positive here by convention
                    agent_position[2] =  -agent_position[2]


                    observations["positions"] = agent_position

                    observations["rotations"] = rotation_r_matrix

                    # rt_4x4 = np.eye(4)
                    # rt_4x4[0:3,0:3] = observations["rotations"]
                    # rt_4x4[0:3,3] = observations["positions"]
                    # rt_4x4_inv = self.safe_inverse_single(rt_4x4)
                    # r, t = self.split_rt_single(rt_4x4_inv)

                    # observations["positions"] = r

                    # observations["positions"] = t

                    # observations["rotations_euler"] = np.array([rx, ry, rz]) #rotation_r.as_euler('xyz', degrees=True)

                    observations["color_sensor"] = rgb
                    observations["depth_sensor"] = event.depth_frame
                    observations["semantic_sensor"] = event.instance_segmentation_frame

                    if False:
                        plt.imshow(rgb)
                        plt_name = f'/home/nel/gsarch/aithor/data/test/img_true{s}{b}.png'
                        plt.savefig(plt_name)
                    
                    cnt += 1 

                    # print("Processed image #", cnt, " for object ", obj['objectType'])

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
    
#     def run(self):
#         event = self.controller.step('GetReachablePositions')
#         self.nav_pts = event.metadata['reachablePositions']
#         self.nav_pts = np.array([list(d.values()) for d in self.nav_pts])
#         np.random.seed(1)
#         # objects = np.random.choice(event.metadata['objects'], self.obj_per_scene, replace=False)
#         objects = event.metadata['objects']
#         objects_inds = np.arange(len(event.metadata['objects']))
#         np.random.shuffle(objects_inds)

#         # objects = np.random.shuffle(event.metadata['objects'])
#         # for obj in event.metadata['objects']: #objects:
#         #     print(obj['name'])
#         # objects = objects[0]
#         successes = 0
#         meta_obj_idx = 0
#         while successes < self.obj_per_scene and meta_obj_idx <= len(event.metadata['objects']) - 1: #obj in objects: #event.metadata['objects']: #objects:
#             obj = objects[objects_inds[meta_obj_idx]]
#             meta_obj_idx += 1
#             print("Object is ", obj['objectType'])
#             # if obj['name'] in ['Microwave_b200e0bc']:
#             #     print(obj['name'])
#             # else:
#             #     continue
#             # print(obj['name'])

#             if obj['objectType'] not in self.include_classes:
#                 print("Continuing... Invalid Object")
#                 continue

#             if not obj['pickupable']:
#                 print("Object ", obj['objectType'], " has no object aligned BB")
#                 continue

            
#             # Calculate distance to object center
#             obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))
#             st()

#             obj_oriented_bb = np.array(list(obj[''].values()))
                        
#             obj_center = np.expand_dims(obj_center, axis=0)
#             distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))

#             # Get points with r_min < dist < r_max
#             valid_pts = self.nav_pts[np.where((distances > self.radius_min)*(distances<self.radius_max))]

#             # Bin points based on angles [vertical_angle (10 deg/bin), horizontal_angle (10 deg/bin)]
#             valid_pts_shift = valid_pts - obj_center

#             dz = valid_pts_shift[:,2]
#             dx = valid_pts_shift[:,0]
#             dy = valid_pts_shift[:,1]

#             # Get yaw for binning 
#             valid_yaw = np.degrees(np.arctan2(dz,dx))

#             nbins = 50
#             bins = np.linspace(-180, 180, nbins+1)
#             bin_yaw = np.digitize(valid_yaw, bins)

#             num_valid_bins = np.unique(bin_yaw).size            

#             if False:
#                 import matplotlib.cm as cm
#                 colors = iter(cm.rainbow(np.linspace(0, 1, nbins)))
#                 plt.figure(2)
#                 plt.clf()
#                 print(np.unique(bin_yaw))
#                 for bi in range(nbins):
#                     cur_bi = np.where(bin_yaw==(bi+1))
#                     points = valid_pts[cur_bi]
#                     x_sample = points[:,0]
#                     z_sample = points[:,2]
#                     plt.plot(z_sample, x_sample, 'o', color = next(colors))
#                 plt.plot(self.nav_pts[:,2], self.nav_pts[:,0], 'x', color='red')
#                 plt.plot(obj_center[:,2], obj_center[:,0], 'x', color = 'black')
#                 plt_name = '/home/nel/gsarch/aithor/data/valid.png'
#                 plt.savefig(plt_name)

#             if num_valid_bins == 0:
#                 continue

#             spawns_per_bin = int(self.num_views / num_valid_bins) + 2
#             # print(f'spawns_per_bin: {spawns_per_bin}')

#             action = "do_nothing"
#             episodes = []
#             valid_pts_selected = []
#             camXs_T_camX0_4x4 = []
#             camX0_T_camXs_4x4 = []
#             origin_T_camXs = []
#             origin_T_camXs_t = []
#             cnt = 0
#             for b in range(nbins):
                
#                 # get all angle indices in the current bin range
#                 inds_bin_cur = np.where(bin_yaw==(b+1)) # bins start 1 so need +1
#                 inds_bin_cur = list(inds_bin_cur[0])
#                 if len(inds_bin_cur) == 0:
#                     continue

#                 for s in range(spawns_per_bin):
#                     observations = {}
                    
#                     if len(inds_bin_cur) == 0:
#                         continue
                    
#                     rand_ind = np.random.randint(0, len(inds_bin_cur))
#                     s_ind = inds_bin_cur.pop(rand_ind)

#                     pos_s = valid_pts[s_ind]
#                     valid_pts_selected.append(pos_s)

#                     # add height from center of agent to camera
#                     pos_s[1] = pos_s[1] + 0.675

#                     # YAW calculation - rotate to object
#                     agent_to_obj = np.squeeze(obj_center) - pos_s 
#                     agent_local_forward = np.array([0, 0, 1.0]) 
#                     flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
#                     flat_dist_to_obj = np.linalg.norm(flat_to_obj)
#                     flat_to_obj /= flat_dist_to_obj

#                     det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
#                     turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

#                     turn_yaw = np.degrees(turn_angle)

#                     turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))

#                     event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1], z=pos_s[2], rotation=dict(x=0.0, y=int(turn_yaw), z=0.0), horizon=int(turn_pitch))

#                     rgb = event.frame

#                     eulers_xyz_rad = np.radians(np.array([event.metadata['agent']['cameraHorizon'], event.metadata['agent']['rotation']['y'], 0.0]))

#                     rx = eulers_xyz_rad[0]
#                     ry = eulers_xyz_rad[1]
#                     rz = eulers_xyz_rad[2]
#                     rotation_r_matrix = self.eul2rotm(-rx, -ry, rz)

#                     agent_position = np.array(list(event.metadata['agent']['position'].values())) + np.array([0.0, 0.675, 0.0])
#                     # need to invert since z is positive here by convention
#                     agent_position[2] =  -agent_position[2]


#                     observations["positions"] = agent_position

#                     observations["rotations"] = rotation_r_matrix

#                     # rt_4x4 = np.eye(4)
#                     # rt_4x4[0:3,0:3] = observations["rotations"]
#                     # rt_4x4[0:3,3] = observations["positions"]
#                     # rt_4x4_inv = self.safe_inverse_single(rt_4x4)
#                     # r, t = self.split_rt_single(rt_4x4_inv)

#                     # observations["positions"] = r

#                     # observations["positions"] = t

#                     # observations["rotations_euler"] = np.array([rx, ry, rz]) #rotation_r.as_euler('xyz', degrees=True)

#                     observations["color_sensor"] = rgb
#                     observations["depth_sensor"] = event.depth_frame
#                     observations["semantic_sensor"] = event.instance_segmentation_frame

#                     if False:
#                         plt.imshow(rgb)
#                         plt_name = f'/home/nel/gsarch/aithor/data/test/img_true{s}{b}.png'
#                         plt.savefig(plt_name)

#                     # print("Processed image #", cnt, " for object ", obj['objectType'])

#                     semantic = event.instance_segmentation_frame
#                     object_id_to_color = event.object_id_to_color
#                     color_to_object_id = event.color_to_object_id

#                     obj_ids = np.unique(semantic.reshape(-1, semantic.shape[2]), axis=0)

#                     instance_masks = event.instance_masks
#                     instance_detections2D = event.instance_detections2D
                    
#                     obj_metadata_IDs = []
#                     for obj_m in event.metadata['objects']: #objects:
#                         obj_metadata_IDs.append(obj_m['objectId'])
                    
#                     object_list = []
#                     for obj_idx in range(obj_ids.shape[0]):
#                         try:
#                             obj_color = tuple(obj_ids[obj_idx])
#                             object_id = color_to_object_id[obj_color]
#                         except:
#                             # print("Skipping ", object_id)
#                             continue

#                         if object_id not in obj_metadata_IDs:
#                             # print("Skipping ", object_id)
#                             continue

#                         obj_meta_index = obj_metadata_IDs.index(object_id)
#                         obj_meta = event.metadata['objects'][obj_meta_index]
#                         obj_category_name = obj_meta['objectType']
                        
#                         # continue if not visible or not in include classes
#                         if obj_category_name not in self.include_classes or not obj_meta['visible']:
#                             continue

#                         obj_instance_mask = instance_masks[object_id]
#                         obj_instance_detection2D = instance_detections2D[object_id] # [start_x, start_y, end_x, end_y]
#                         obj_instance_detection2D = np.array([obj_instance_detection2D[1], obj_instance_detection2D[0], obj_instance_detection2D[3], obj_instance_detection2D[2]])  # ymin, xmin, ymax, xmax

#                         if False:
#                             print(object_id)
#                             plt.imshow(obj_instance_mask)
#                             plt_name = f'/home/nel/gsarch/aithor/data/test/img_mask{s}.png'
#                             plt.savefig(plt_name)

#                         obj_center_axisAligned = np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))
#                         obj_center_axisAligned[2] = -obj_center_axisAligned[2]
#                         obj_size_axisAligned = np.array(list(obj_meta['axisAlignedBoundingBox']['size'].values()))
                        
#                         # print(obj_category_name)

#                         if self.verbose: 
#                             print("Saved class name is : ", obj_category_name)

#                         obj_data = {'instance_id': object_id, 'category_id': object_id, 'category_name': obj_category_name,
#                                         'bbox_center': obj_center_axisAligned, 'bbox_size': obj_size_axisAligned,
#                                             'mask_2d': obj_instance_mask, 'box_2d': obj_instance_detection2D}
#                         # object_list.append(obj_instance)
#                         object_list.append(obj_data)
                    
#                     observations["object_list"] = object_list

#                     # check if object visible (make sure agent is not behind a wall)
#                     obj_id = obj['objectId']
#                     obj_id_to_color = object_id_to_color[obj_id]
#                     # if np.sum(obj_ids==object_id_to_color[obj_id]) > 0:
#                     if self.verbose:
#                         print("episode is valid......")
#                     episodes.append(observations)
#                     # print(cnt)

#                     if False:
#                         if cnt > 0:

#                             origin_T_camX = episodes[cnt]["rotations"]

#                             r = R.from_matrix(origin_T_camX)
#                             print("EULER CHECK: ", r.as_euler('xyz', degrees=True))

#                             camX0_T_camX = np.matmul(camX0_T_origin, origin_T_camX)
#                             r = R.from_matrix(camX0_T_camX)
#                             print("EULER CHECK: ", r.as_euler('xyz', degrees=True))
#                             # camX0_T_camX = np.matmul(camX0_T_origin, origin_T_camX)
#                             r = R.from_matrix(camX0_T_origin)
#                             print("EULER CHECK: ", r.as_euler('xyz', degrees=True))

#                             origin_T_camXs.append(origin_T_camX)
#                             origin_T_camXs_t.append(episodes[cnt]["positions"])

#                             origin_T_camX_4x4 = np.eye(4)
#                             origin_T_camX_4x4[0:3, 0:3] = origin_T_camX
#                             origin_T_camX_4x4[:3,3] = episodes[cnt]["positions"]
#                             camX0_T_camX_4x4 = np.matmul(camX0_T_origin_4x4, origin_T_camX_4x4)
#                             camX_T_camX0_4x4 = self.safe_inverse_single(camX0_T_camX_4x4)

#                             camXs_T_camX0_4x4.append(camX_T_camX0_4x4)
#                             camX0_T_camXs_4x4.append(camX0_T_camX_4x4)

#                             camX0_T_camX_quat = quaternion.from_rotation_matrix(camX0_T_camX)
#                             camX0_T_camX_eul = quaternion.as_euler_angles(camX0_T_camX_quat)

#                             camX0_T_camX_4x4 = self.safe_inverse_single(camX_T_camX0_4x4)
#                             origin_T_camX_4x4 = np.matmul(origin_T_camX0_4x4, camX0_T_camX_4x4)
#                             r_origin_T_camX, t_origin_T_camX, = self.split_rt_single(origin_T_camX_4x4)

#                             if self.verbose:
#                                 print(r_origin_T_camX)
#                                 print(origin_T_camX)

#                         else:
#                             origin_T_camX0 = episodes[0]["rotations"]
#                             camX0_T_origin = np.linalg.inv(origin_T_camX0)
#                             # camX0_T_origin = self.safe_inverse_single(origin_T_camX0)

#                             origin_T_camXs.append(origin_T_camX0)
#                             origin_T_camXs_t.append(episodes[0]["positions"])

#                             origin_T_camX0_4x4 = np.eye(4)
#                             origin_T_camX0_4x4[0:3, 0:3] = origin_T_camX0
#                             origin_T_camX0_4x4[:3,3] = episodes[0]["positions"]
#                             camX0_T_origin_4x4 = self.safe_inverse_single(origin_T_camX0_4x4)

#                             camXs_T_camX0_4x4.append(np.eye(4))

#                             camX0_T_camXs_4x4.append(np.eye(4))

#                             origin_T_camX0_t = episodes[0]["positions"]
                    
#                     cnt +=1

                
#             if len(episodes) >= self.num_views:
#                 print(f'num episodes: {len(episodes)}')
#                 data_folder = obj['name']
#                 data_path = os.path.join(self.basepath, data_folder)
#                 print("Saving to ", data_path)
#                 os.mkdir(data_path)
#                 # flat_obs = np.random.choice(episodes, self.num_views, replace=False)
#                 np.random.seed(1)
#                 rand_inds = np.sort(np.random.choice(len(episodes), self.num_views, replace=False))
#                 bool_inds = np.zeros(len(episodes), dtype=bool)
#                 bool_inds[rand_inds] = True
#                 flat_obs = np.array(episodes)[bool_inds]
#                 flat_obs = list(flat_obs)
#                 viewnum = 0
#                 if False:
#                     self.generate_xyz_habitatCamXs(flat_obs)
#                 for obs in flat_obs:
#                     self.save_datapoint(obs, data_path, viewnum, True)
#                     viewnum += 1
#                 print("SUCCESS #", successes)
#                 successes += 1
#             else:
#                 print("Not enough episodes:", len(episodes))

if __name__ == '__main__':
    Ai2Thor()

