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


        random.shuffle(mapnames)
        self.mapnames = mapnames
        self.num_episodes = 1 #len(self.mapnames)   

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

        self.small_classes = []
        self.rot_interval = 5.0
        self.radius_max = 3.5 #3 #1.75
        self.radius_min = 1.0 #1.25
        self.num_flat_views = 3
        self.num_any_views = 7
        self.num_views = 25

        self.obj_per_scene = 10

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

        self.fov = 90
        hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2.
        self.pix_T_camX[1,2] = self.H/2.


        self.run_episodes()

    def run_episodes(self):
        self.ep_idx = 0
        # self.objects = []
        
        for episode in range(self.num_episodes):
            print("STARTING EPISODE ", episode)

            mapname = self.mapnames[episode]
            print("MAPNAME=", mapname)

            self.controller = Controller(
                scene=mapname, 
                gridSize=0.25,
                width=self.W,
                height=self.H,
                fieldOfView= self.fov,
                renderObjectImage=True,
                renderDepthImage=True,
                )
            
            self.basepath = self.homepath + f"/{mapname}_{episode}"
            print("BASEPATH: ", self.basepath)

            # self.basepath = f"/hdd/ayushj/habitat_data/{mapname}_{episode}"
            if not os.path.exists(self.basepath):
                os.mkdir(self.basepath)

            self.run()
            
            self.controller.stop()
            time.sleep(1)

            self.ep_idx += 1

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
    
    def run2(self):
        event = self.controller.step('GetReachablePositions')
        for obj in event.metadata['objects']:
            if obj['objectType'] not in self.objects:
                self.objects.append(obj['objectType'])
    
    def run(self):
        event = self.controller.step('GetReachablePositions')
        self.nav_pts = event.metadata['reachablePositions']
        self.nav_pts = np.array([list(d.values()) for d in self.nav_pts])
        objects = np.random.choice(event.metadata['objects'], self.obj_per_scene, replace=False)

        # objects = np.random.shuffle(event.metadata['objects'])
        # for obj in event.metadata['objects']: #objects:
        #     print(obj['name'])
        # objects = objects[0]
        for obj in objects:
            print("Object is ", obj['objectType'])
            # if obj['name'] in ['Microwave_b200e0bc']:
            #     print(obj['name'])
            # else:
            #     continue
            # print(obj['name'])

            if obj['objectType'] not in self.include_classes:
                print("Continuing... Invalid Object")
                continue

            
            # Calculate distance to object center
            # obj_center = np.array(list(obj['position'].values()))
            obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))
            
            # obj_center = np.array(list(obj['position'].values()))
            
            #print(obj_center)
            obj_center = np.expand_dims(obj_center, axis=0)
            #print(obj_center)
            distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))

            # Get points with r_min < dist < r_max
            valid_pts = self.nav_pts[np.where((distances > self.radius_min)*(distances<self.radius_max))]
            # if not valid_pts:
                # continue

            # plot valid points that we happen to select
            # self.plot_navigable_points(valid_pts)

            # Bin points based on angles [vertical_angle (10 deg/bin), horizontal_angle (10 deg/bin)]
            valid_pts_shift = valid_pts - obj_center

            dz = valid_pts_shift[:,2]
            dx = valid_pts_shift[:,0]
            dy = valid_pts_shift[:,1]

            # Get yaw for binning 
            valid_yaw = np.degrees(np.arctan2(dz,dx))

            # # pitch calculation 
            # dxdz_norm = np.sqrt((dx * dx) + (dz * dz))
            # valid_pitch = np.degrees(np.arctan2(dy,dxdz_norm))

            # binning yaw around object
            # nbins = 18

            nbins = 18
            bins = np.linspace(-180, 180, nbins+1)
            bin_yaw = np.digitize(valid_yaw, bins)

            num_valid_bins = np.unique(bin_yaw).size

            # spawns_per_bin = int(self.num_views / num_valid_bins) + 2
            

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

            spawns_per_bin = int(self.num_views / num_valid_bins) + 2
            # print(f'spawns_per_bin: {spawns_per_bin}')

            action = "do_nothing"
            episodes = []
            valid_pts_selected = []
            cnt = 0
            for b in range(nbins):
                
                # get all angle indices in the current bin range
                # st()
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

                    # st()
                    # s_ind = np.random.choice(inds_bin_cur)
                    #s_ind = inds_bin_cur[0][0]
                    pos_s = valid_pts[s_ind]
                    valid_pts_selected.append(pos_s)

                    # event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1], z=pos_s[2], rotation=dict(x=0.0, y=180.0, z=0.0), horizon=0.0)
                    # agent_pos = list(event.metadata['agent']['position'].values())
                    # print("Agent rot " , event.metadata['agent']['rotation'])
                    # print("Agent pos " , event.metadata['agent']['position'])
                    # print("Object center ", obj['axisAlignedBoundingBox']['center'])

                    # add height from center of agent to camera
                    pos_s[1] = pos_s[1] + 0.675

                    # YAW calculation - rotate to object
                    agent_to_obj = np.squeeze(obj_center) - pos_s # + np.array([0.0, 0.675, 0.0]))
                    agent_local_forward = np.array([0, 0, -1.0]) # y, z, x
                    # agent_local_forward = np.array([-1.0, 0, 0]) # y, z, x
                    flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                    flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                    flat_to_obj /= flat_dist_to_obj

                    det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
                    turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
                    # quat_yaw = self.quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))

                    # turn_yaw = np.degrees(quaternion.as_euler_angles(quat_yaw)[1])
                    turn_yaw = np.degrees(turn_angle)
                    # print("Turn Yaw=", turn_yaw)
                    # print("turn1 beg ", turn_yaw)

                    # agent_pos = list(event.metadata['agent']['position'].values())
                    # dist_to_origin = np.sqrt(agent_pos[0]**2 + agent_pos[2]**2)
                    # dist_to_object = np.sqrt((agent_pos[0] - obj_center[0,0])**2 + (agent_pos[2] - obj_center[0,2])**2)

                    # print("Agent rot " , event.metadata['agent']['rotation'])

                    # p0 = agent_pos
                    # p1 = np.squeeze(obj_center)
                    # C = np.cross(p0, p1) 
                    # D = np.dot(p0, p1)
                    # NP0 = np.linalg.norm(p0) 
                    # if ~np.all(C==0): # check for colinearity    
                    #     Z = np.array([[0, -C[2], C[1]], [C[2], 0, -C[0]], [-C[1], C[0], 0]])
                    #     R = (np.eye(3) + Z + Z**2 * (1-D)/(np.linalg.norm(C)**2)) / NP0**2 # rotation matrix
                    # else:
                    #     R = np.sign(D) * (np.linalg.norm(p1) / NP0) # orientation and scaling
                    
                    # quat_rot = quaternion.from_rotation_matrix(R)
                    # turns = np.degrees(quaternion.as_euler_angles(quat_rot))
                    # turn_yaw = turns[1]
                    # print("turn1 ", turn_yaw)
                    # print("turn0 ", turns[0])
                    # print("turn2 ", turns[2])

                    # if dist_to_origin > dist_to_object:
                    #     print('or>ob')
                    #     turn_yaw2 = np.degrees(np.cos(dist_to_object/dist_to_origin))
                    # else:
                    #     print('or>ob')
                    #     turn_yaw2 = np.degrees(np.cos(dist_to_origin/dist_to_object))
                    # print("TURN YAW2, ", turn_yaw2)
                    # Calculate Pitch from head to object
                    turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))
                    # movement = "LookUp" if turn_pitch>0 else "LookDown"
                    # event = controller.step(movement, degrees=np.abs(turn_pitch))

                    event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1], z=pos_s[2], rotation=dict(x=0.0, y=180.0 + int(turn_yaw), z=0.0), horizon=int(turn_pitch))
                    # movement = "RotateRight" if turn_yaw>0 else "RotateLeft"
                    # event = self.controller.step(action='RotateRight', rotation=int(np.abs(turn_yaw)))

                    # movement = "LookDown" if turn_pitch>0 else "LookUp"
                    # event = self.controller.step(movement, degrees=np.abs(turn_pitch))

                    # print("Agent rot " , event.metadata['agent']['rotation'])

                    # angle_ranges = np.arange(0, 360, 15)
                    # angles_test = np.ones_like(angle_ranges) * 15
                    # for i in angle_ranges:
                    #     movement = "RotateRight"
                    #     event = self.controller.step(movement, degrees=15.0)

                    #     rgb = event.frame
                        
                    #     if True:
                    #         plt.imshow(rgb)
                    #         plt_name = f'/home/nel/gsarch/aithor/data/img{i}.png'.format(i)
                    #         plt.savefig(plt_name)
                    # print(event.metadata['agent']['position'])
                    # print(event.metadata['agent']['rotation'])

                    rgb = event.frame

                    rotation_euler_radians = np.radians(np.array([event.metadata['agent']['cameraHorizon'], event.metadata['agent']['rotation']['y'], 0.0]))

                    observations["positions"] = np.array(list(event.metadata['agent']['position'].values())) + np.array([0.0, 0.675, 0.0])
                    # print(observations["positions"])
                    # print(pos_s)
                    # observations["rotations"] = quaternion.from_euler_angles(np.radians(np.array(list(event.metadata['agent']['rotation'].values()))))
                    observations["rotations"] = quaternion.from_euler_angles(rotation_euler_radians)

                    # print(observations["positions"])

                    observations["color_sensor"] = rgb
                    observations["depth_sensor"] = event.depth_frame
                    observations["semantic_sensor"] = event.instance_segmentation_frame

                    if False:
                        plt.imshow(rgb)
                        plt_name = f'/home/nel/gsarch/aithor/data/test/img_true{s}{b}.png'
                        plt.savefig(plt_name)

                    # print("Processed image #", cnt, " for object ", obj['objectType'])

                    semantic = event.instance_segmentation_frame
                    object_id_to_color = event.object_id_to_color
                    color_to_object_id = event.color_to_object_id

                    obj_ids = np.unique(semantic.reshape(-1, semantic.shape[2]), axis=0)

                    instance_masks = event.instance_masks
                    instance_detections2D = event.instance_detections2D
                    
                    obj_metadata_IDs = []
                    for obj_m in event.metadata['objects']: #objects:
                        obj_metadata_IDs.append(obj_m['objectId'])
                    
                    object_list = []
                    for obj_idx in range(obj_ids.shape[0]):
                        try:
                            obj_color = tuple(obj_ids[obj_idx])
                            object_id = color_to_object_id[obj_color]
                        except:
                            # print("Skipping ", object_id)
                            continue

                        if object_id not in obj_metadata_IDs:
                            # print("Skipping ", object_id)
                            continue

                        obj_meta_index = obj_metadata_IDs.index(object_id)
                        obj_meta = event.metadata['objects'][obj_meta_index]
                        obj_category_name = obj_meta['objectType']
                        
                        # continue if not visible or not in include classes
                        if obj_category_name not in self.include_classes or not obj_meta['visible']:
                            continue

                        obj_instance_mask = instance_masks[object_id]
                        obj_instance_detection2D = instance_detections2D[object_id] # [start_x, start_y, end_x, end_y]
                        obj_instance_detection2D = np.array([obj_instance_detection2D[1], obj_instance_detection2D[0], obj_instance_detection2D[3], obj_instance_detection2D[2]])  # ymin, xmin, ymax, xmax

                        if False:
                            print(object_id)
                            plt.imshow(obj_instance_mask)
                            plt_name = f'/home/nel/gsarch/aithor/data/test/img_mask{s}.png'
                            plt.savefig(plt_name)

                        obj_center_axisAligned = np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))
                        obj_size_axisAligned = np.array(list(obj_meta['axisAlignedBoundingBox']['size'].values()))
                        
                        # print(obj_category_name)

                        if self.verbose: 
                            print("Saved class name is : ", obj_category_name)

                        obj_data = {'instance_id': object_id, 'category_id': object_id, 'category_name': obj_category_name,
                                        'bbox_center': obj_center_axisAligned, 'bbox_size': obj_size_axisAligned,
                                            'mask_2d': obj_instance_mask, 'box_2d': obj_instance_detection2D}
                        # object_list.append(obj_instance)
                        object_list.append(obj_data)
                    
                    observations["object_list"] = object_list

                    # check if object visible (make sure agent is not behind a wall)
                    obj_id = obj['objectId']
                    obj_id_to_color = object_id_to_color[obj_id]
                    # if np.sum(obj_ids==object_id_to_color[obj_id]) > 0:
                    if self.verbose:
                        print("episode is valid......")
                    episodes.append(observations)
                    
                    cnt +=1
                
            if len(episodes) >= self.num_views:
                print(f'num episodes: {len(episodes)}')
                data_folder = obj['name']
                data_path = os.path.join(self.basepath, data_folder)
                print("Saving to ", data_path)
                os.mkdir(data_path)
                # flat_obs = np.random.choice(episodes, self.num_views, replace=False)
                rand_inds = np.sort(np.random.choice(len(episodes), self.num_views, replace=False))
                bool_inds = np.zeros(len(episodes), dtype=bool)
                bool_inds[rand_inds] = True
                flat_obs = np.array(episodes)[bool_inds]
                flat_obs = list(flat_obs)
                viewnum = 0
                for obs in flat_obs:
                    self.save_datapoint(obs, data_path, viewnum, True)
                    viewnum += 1
            else:
                print("Not enough episodes:", len(episodes))
                    


                    # mainobj_id = obj.id
                    # main_id = int(mainobj_id[1:])
                    # semantic = observations["semantic_sensor"]
                    # mask = np.zeros_like(semantic)
                    # mask[semantic == main_id] = 1
                    # obj_is_visible = False if np.sum(mask)==0 else True

                    # if self.is_valid_datapoint(observations, obj) and obj_is_visible:
                    #     if self.verbose:
                    #         print("episode is valid......")
                    #     episodes.append(observations)
                    #     if self.visualize:
                    #         rgb = observations["color_sensor"]
                    #         semantic = observations["semantic_sensor"]
                    #         depth = observations["depth_sensor"]
                    #         self.display_sample(rgb, semantic, depth, mainobj=obj, visualize=False)
                    # if self.visualize:
                    #         rgb = observations["color_sensor"]
                    #         semantic = observations["semantic_sensor"]
                    #         depth = observations["depth_sensor"]
                    #         self.display_sample(rgb, semantic, depth, mainobj=obj, visualize=False)

                    #print("agent_state: position", self.agent.state.position, "rotation", self.agent.state.rotation)

                    
        

if __name__ == '__main__':
    Ai2Thor()

controller = Controller(scene='FloorPlan28', gridSize=0.25)

# event = controller.step(action='MoveAhead')
# for obj in controller.last_event.metadata['objects']:
#     print(obj['objectId'])

event = controller.step('GetReachablePositions')
positions = event.metadata['reachablePositions']

angle = np.random.uniform(0, 359)
event = controller.step(action='Rotate', rotation=angle)

# for o in event.metadata['objects']:
#     print(o['objectId'])

obj_cur = np.random.choice(event.metadata['objects'])
print(obj_cur['objectId'])
obj_pos = list(obj_cur['position'].keys())
obj_rot = list(obj_cur['rotation'].keys())

event = controller.step('TeleportFull', x=obj_pos[0], y=obj_pos[1], z=obj_pos[2], rotation=dict(x=obj_rot[0], y=obj_rot[1], z=obj_rot[2]), horizon=30.0)
im_rgb = event.frame #cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
# im_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

plt.imshow(im_rgb)
plt_name = '/home/nel/gsarch/aithor/data/' + 'img.png'
plt.savefig(plt_name)

# cv2.imshow('image',bgr)
# cv2.waitKey(0)

