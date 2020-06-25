import fnmatch
import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch

class BackgroundDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """
    
    def __init__(self, img_dir, imgsize, shuffle=True):
        n_jpeg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpeg'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_jpeg_images + n_jpg_images
        self.len = n_images
        self.img_dir = img_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.jpeg') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        
        image = self.scale(image)
        transform = transforms.ToTensor()
        image = transform(image)
        return image
    
    def scale(self, img):
        """

        Args:
            img:

        Returns:

        """
        
        w, h = img.size
        if w == h:
            scaled_img = img
        else:
            dim_to_scale = 1 if w < h else 2
            if dim_to_scale == 1:
                cropping = (h - w) / 2
                scaled_img = img.crop((0, int(cropping), w, int(cropping) + w))
            
            else:
                cropping = (w - h) / 2
                scaled_img = img.crop((int(cropping), 0, int(cropping) + h, h))
        
        resize = transforms.Resize((self.imgsize, self.imgsize))
        
        scaled_img = resize(scaled_img)  # choose here
        
        return scaled_img


# class MeshDataset(Dataset):
#     """InriaDataset: representation of the INRIA person dataset.
#
#     Internal representation of the commonly used INRIA person dataset.
#     Available at: http://pascal.inrialpes.fr/data/human/
#
#     Attributes:
#         len: An integer number of elements in the
#         img_dir: Directory containing the images of the INRIA dataset.
#         lab_dir: Directory containing the labels of the INRIA dataset.
#         mesh_names: List of all image file names in img_dir.
#         shuffle: Whether or not to shuffle the dataset.
#
#     """
#
#     def __init__(self, mesh_dir, shuffle=True):
#         n_mesh = len(fnmatch.filter(os.listdir(mesh_dir), '*.pkl'))
#         # n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
#         # n_images = n_jpeg_images + n_jpg_images
#         self.len = n_mesh
#         self.mesh_dir = mesh_dir
#         # self.imgsize = imgsize
#         self.mesh_names = fnmatch.filter(os.listdir(mesh_dir), '*.pkl')
#         # self.logo_names = fnmatch.filter(os.listdir(mesh_dir), '*.pickle')
#         self.mesh_paths = []
#         self.shuffle = shuffle
#         # self.logo_paths = []
#         for mesh_name in self.mesh_names:
#             self.mesh_paths.append(os.path.join(self.mesh_dir, mesh_name))
#         # for logo_name in self.logo_names:
#         #     self.logo_paths.append(os.path.join(self.mesh_dir, logo_name))
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         assert idx <= len(self), 'index range error'
#         mesh = torch.load(self.mesh_paths[idx])
#         vertices = mesh['vertices']
#         faces = mesh['faces']
#         textures = mesh['textures'].unsqueeze(0)
#         logo_indexs = mesh['logo_indexs']
#         logo_scale = mesh['logo_scale']
#         # return mesh
#         # mesh_path = os.path.join(self.mesh_dir, self.mesh_names[idx])
#         # vertices, faces, textures = nr.load_obj(mesh_path, load_texture=True)
#         # with open(self.logo_names[idx],'rb') as file:
#         #     logo = np.array(pickle.load(file))
#         # vertices = vertices.unsqueeze(0)
#         # faces = faces.unsqueeze(0)
#         # textures = textures.unsqueeze(0)
#         return vertices, faces, textures, logo_indexs, logo_scale

# class MeshDataset():
#     """InriaDataset: representation of the INRIA person dataset.
#
#     Internal representation of the commonly used INRIA person dataset.
#     Available at: http://pascal.inrialpes.fr/data/human/
#
#     Attributes:
#         len: An integer number of elements in the
#         img_dir: Directory containing the images of the INRIA dataset.
#         lab_dir: Directory containing the labels of the INRIA dataset.
#         mesh_names: List of all image file names in img_dir.
#         shuffle: Whether or not to shuffle the dataset.
#
#     """
#
#     def __init__(self, mesh_dir):
#         n_mesh = len(fnmatch.filter(os.listdir(mesh_dir), '*.pkl'))
#         # n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
#         # n_images = n_jpeg_images + n_jpg_images
#         self.len = n_mesh
#         self.mesh_dir = mesh_dir
#         # self.imgsize = imgsize
#         self.mesh_names = fnmatch.filter(os.listdir(mesh_dir), '*.pkl')
#         # self.logo_names = fnmatch.filter(os.listdir(mesh_dir), '*.pickle')
#         self.mesh_paths = []
#         # self.logo_paths = []
#         for mesh_name in self.mesh_names:
#             self.mesh_paths.append(os.path.join(self.mesh_dir, mesh_name))
#         # for logo_name in self.logo_names:
#         #     self.logo_paths.append(os.path.join(self.mesh_dir, logo_name))
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         assert idx <= len(self), 'index range error'
#         mesh = torch.load(self.mesh_paths[idx])
#         vertices = mesh['vertices']
#         faces = mesh['faces']
#         textures = mesh['textures'].unsqueeze(0)
#         logo_indexs = mesh['logo_indexs']
#         logo_scale = mesh['logo_scale']
#         # return mesh
#         # mesh_path = os.path.join(self.mesh_dir, self.mesh_names[idx])
#         # vertices, faces, textures = nr.load_obj(mesh_path, load_texture=True)
#         # with open(self.logo_names[idx],'rb') as file:
#         #     logo = np.array(pickle.load(file))
#         # vertices = vertices.unsqueeze(0)
#         # faces = faces.unsqueeze(0)
#         # textures = textures.unsqueeze(0)
#         return vertices, faces, textures, logo_indexs, logo_scale
    


# data_dir = '//home/zhouge/Documents/3dmesh/pictures/adv_data/downloads1/avenue'
# batch_size = 32
# num_workers = 8
# img_size = 416
# train_loader = DataLoader(BackgroundDataset(data_dir, img_size,
#                                             shuffle=True),
#                           batch_size=batch_size,
#                           shuffle=True,
#                           num_workers=10)
#
# for i_batch, data in enumerate(train_loader):
#     print(data.shape)
