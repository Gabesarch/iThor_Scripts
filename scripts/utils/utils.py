import numpy as np
import quaternion
from argparse import Namespace
import math


class Utils():
    def __init__(self, fov, W, H):  
        self.H = H
        self.W = W
        self.fov = fov
        self.K = self.get_habitat_pix_T_camX(self.fov)


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
        
        sy = np.sqrt(r00*r00 + r10*r10)
        
        cond = (sy > 1e-6)
        rx = np.where(cond, np.arctan2(r21, r22), np.arctan2(-r12, r11))
        ry = np.where(cond, np.arctan2(-r20, sy), np.arctan2(-r20, sy))
        rz = np.where(cond, np.arctan2(r10, r00), np.zeros_like(r20))

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

    
    def save_datapoint(self, observations, data_path, viewnum, flat_view):
        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        # st()
        depth = observations["depth_sensor"]
        agent_pos = observations["positions"] 
        agent_rot = observations["rotations"]
        # Assuming all sensors have same extrinsics
        color_sensor_pos = observations["positions"] 
        color_sensor_rot = observations["rotations"] 
        object_list = observations['object_list']


        if False:
            plt.imshow(rgb)
            plt_name = f'/home/nel/gsarch/aithor/data/test/img_mask{viewnum}.png'
            plt.savefig(plt_name)
                
        save_data = {'flat_view': flat_view, 'objects_info': object_list,'rgb_camX':rgb, 'depth_camX': depth, 'sensor_pos': color_sensor_pos, 'sensor_rot': color_sensor_rot}
        
        with open(os.path.join(data_path, str(viewnum) + ".p"), 'wb') as f:
            pickle.dump(save_data, f)
        f.close()
