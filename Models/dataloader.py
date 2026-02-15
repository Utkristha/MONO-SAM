import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes,NuScenesExplorer
from nuscenes.utils.geometry_utils import view_points
from torch.utils.data import Dataset,DataLoader
data_path = 'D:/Projects/Minor Project/Datasets/raw/nuScenes'


class CustomDataset(Dataset):
    def __init__(self, nusc, explorer, sample_list):
        self.nusc = nusc
        self.explorer = explorer
        self.samples = sample_list
        # Scaling factors for 64x64 depth map
        self.scaled_u = 64/1600
        self.scaled_v = 64/900
        self.max_dist = 80.0 # Standard max distance for nuScenes LiDAR

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        cam_token = sample['data']['CAM_FRONT']
        lidar_token = sample['data']['LIDAR_TOP']

        # 1. Get Image
        # map_pointcloud_to_image returns the image and projected lidar points
        points, coloring, image = self.explorer.map_pointcloud_to_image(lidar_token, cam_token)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

        # 2. Generate Ground Truth Segmentation Mask (900x1600)
        mask = np.zeros((900, 1600), dtype=np.float32)
        
        # Get boxes and camera intrinsics for projection
        _, boxes, camera_intrinsic = self.nusc.get_sample_data(cam_token)
        
        for box in boxes:
            # Focus on vehicles for this task
            if 'vehicle' in box.name:
                # Project 3D corners [3, 8] to 2D pixel coordinates [2, 8]
                pts_2d = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
                pts_2d = pts_2d.T.astype(np.int32)
                
                # Create a solid polygon from the projected 8 corners
                hull = cv2.convexHull(pts_2d)
                cv2.fillPoly(mask, [hull], 1.0)

        # Scale mask to 256x256 to match the SAM decoder's internal loss resolution
        mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 3. Generate Ground Truth Depth Map (64x64)
        # Channel 0: Normalized distance, Channel 1: Validity mask (1 if point exists)
        depth_map = np.zeros((2, 64, 64), dtype=np.float32)

        u_cord = points[0, :]
        v_cord = points[1, :]
        
        for i in range(len(coloring)):
            u = int(u_cord[i] * self.scaled_u)
            v = int(v_cord[i] * self.scaled_v)

            if 0 <= u < 64 and 0 <= v < 64:
                # Normalize distance by max_dist (0.0 to 1.0 range)
                norm_dist = min(coloring[i] / self.max_dist, 1.0)
                
                # Keep the closest point for each 64x64 cell
                if depth_map[0, v, u] == 0 or norm_dist < depth_map[0, v, u]:
                    depth_map[0, v, u] = norm_dist
                    depth_map[1, v, u] = 1.0 # Mark as a valid ground truth pixel

        return {
            "image": image_tensor,                              # [3, 900, 1600]
            "gt_depth": torch.from_numpy(depth_map),            # [2, 64, 64]
            "gt_mask": torch.from_numpy(mask_resized).unsqueeze(0), # [1, 256, 256]
            "sample_token": sample['token']
        }

if __name__ == '__main__':
    nusc = NuScenes(version="v1.0-mini",dataroot=data_path,verbose=True)
    explorer = NuScenesExplorer(nusc)

    dataset = CustomDataset(nusc,explorer,nusc.sample)
