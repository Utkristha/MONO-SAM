import numpy as np
from nuscenes.nuscenes import NuScenes,NuScenesExplorer
import torch
from torch.utils.data import Dataset,DataLoader
data_path = 'D:/Projects/Minor Project/Datasets/raw/nuScenes'

class CustomDataset(Dataset):
    
    def __init__(self,nusc,explorer,sample_list):
        self.nusc = nusc
        self.explorer = explorer
        self.samples = sample_list
        self.scaled_u = 64/1600
        self.scaled_v = 64/900

    
    def __len__(self): 
        return len(self.samples)

    def __getitem__(self,index):
        sample = self.samples[index]

        cam_token = sample['data']['CAM_FRONT']
        lidar_token = sample['data']['LIDAR_TOP']

        points,coloring,image = self.explorer.map_pointcloud_to_image(lidar_token,cam_token)

        image_tensor = torch.from_numpy(np.array(image)).permute(2,0,1).float()

        depth_map = np.zeros((2,64,64),dtype=np.float32)

        u_cord = points[0,:]
        v_cord = points[1,:]

        max_dist = 80
        
        for i in range(len(coloring)):
            u = int(u_cord[i] * self.scaled_u)
            v = int(v_cord[i] * self.scaled_v)

            if 0 <= u < 64 and 0 <= v < 64:
                if depth_map[0, v, u] == 0 or coloring[i] < depth_map[0, v, u]:
                    depth_map[0, v, u] = min(coloring[i] / max_dist, 1.0)
                    depth_map[1,v,u] = 1.0

        return {
            "depth_map" : torch.from_numpy(depth_map),
            "sample_token" : sample['token'],
            "image" : image_tensor 
        }

if __name__ == '__main__':
    nusc = NuScenes(version="v1.0-mini",dataroot=data_path,verbose=True)
    explorer = NuScenesExplorer(nusc)

    dataset = CustomDataset(nusc,explorer,nusc.sample)
