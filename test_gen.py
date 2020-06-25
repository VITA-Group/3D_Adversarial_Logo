import os

import cv2
from PIL import Image
import imageio
import numpy as np
import torch
import torch.nn.functional
import torchvision.transforms as transforms
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
logo_index_path = os.path.join(data_dir, 'logo_index.pickle')
# with open(logo_index_path, 'rb') as logo_file:
#     logo_indexs = np.array(pickle.load(logo_file))
street_img_path = os.path.join(data_dir, 'street.jpg')
img_path = os.path.join(data_dir, 'example1.jpg')


class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref, filename_logo):
        super(Model, self).__init__()
        vertices, faces, textures = nr.load_obj(filename_obj, load_texture=True)
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])
        textures = textures[None, :, :, :, :, :]
        self.grad_textures = nn.Parameter(torch.full(textures.size(), 0.5).cuda())
        
        # load reference image
        with open(filename_logo, 'rb') as logo_file:
            logo_indexs = np.array(pickle.load(logo_file))
        textures[:, logo_indexs, :, :, :, :] = self.grad_textures[:, logo_indexs, :, :, :, :]
        self.textures = textures
        image_ref = Image.open(filename_ref).convert('RGB')
        self.image_ref = image_ref
        
        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer
    
    def forward(self, img_size, batch_size, i_batch, angle_range):
        # self.renderer.eye = nr.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        start = 140 + i_batch * angle_range
        end = start + angle_range
        loop = tqdm(range(start, end, 4))
        training_images = []
        for num, azimuth in enumerate(loop):
            loop.set_description('Drawing')
            self.renderer.eye = nr.get_points_from_angles(2.0, 0., azimuth)
            image, _, _ = self.renderer(self.vertices, self.faces,
                                        self.textures)  # [batch_size, RGB, image_size, image_size]
            image = self.pad(image, self.image_ref, img_size)
            training_images.append(image)
        
        return torch.cat(training_images, 0)
    
    def pad(self, img, image_ref, img_size):
        # street_img = street_img.transpose(1, 2, 0)
        
        w, h = image_ref.size
        if w == h:
            padded_img = image_ref
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(image_ref, (int(padding), 0))
            
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(image_ref, (0, int(padding)))
        transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor()])
        
        image_ref = transform(padded_img).unsqueeze(0)
        
        h, w = img.shape[2:]
        for i in range(h):
            for j in range(w):
                if not img[0, 2, i, j] == 0.:
                    image_ref[:, :, 85 + i, 80 + j] = img[:, :, i, j]
        
        return image_ref

def people_load():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    filename_input = os.path.join(data_dir, 'human.obj')
    # load .obj
    # vertices, faces, textures = nr.load_obj(args.filename_input, load_texture=True)
    # street_img = imageio.imread(street_img_path)
    vertices, faces, textures = nr.load_obj(filename_input, load_texture=True)
    
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    textures = textures[None, :, :, :, :, :]
    
    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    # textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    # textures[:, logo_indexs, :, :, :, 0] = grad_textures[:, logo_indexs, :, :, :, 0]
    # to gpu
    return vertices, faces, textures
    # print(textures.shape)
    # create renderer


def generate_data(vertices, faces, textures, batch_size, img_size, grad_textures):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-wr', '--angle_range', type=int, default=40)
    # args = parser.parse_args()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    street_img_path = os.path.join(data_dir, 'street.jpg')
    street_img = Image.open(street_img_path).convert('RGB')
    logo_index_path = os.path.join(data_dir, 'logo_index.pickle')
    with open(logo_index_path, 'rb') as logo_file:
        logo_indexs = np.array(pickle.load(logo_file))
    renderer = nr.Renderer(camera_mode='look_at')
    camera_distance = 2.0
    elevation = 0
    
    def pad(img, street_img, imgsize):
        # street_img = street_img.transpose(1, 2, 0)
        
        w, h = street_img.size
        if w == h:
            padded_img = street_img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(street_img, (int(padding), 0))
            
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(street_img, (0, int(padding)))
        transform = transforms.Compose([transforms.Resize((imgsize, imgsize)),
                                        transforms.ToTensor()])
        
        street_img = transform(padded_img).unsqueeze(0)
        
        h, w = img.shape[2:]
        for i in range(h):
            for j in range(w):
                if not img[0, 2, i, j] == 0.:
                    street_img[:, :, 85 + i, 80 + j] = img[:, :, i, j]
        
        return street_img
    
    grad_textures = grad_textures.cuda()
    textures[:, logo_indexs, :, :, :, :] = grad_textures[:, logo_indexs, :, :, :, :]
    # draw object
    angle_range = 4
    loop = tqdm(range(180 - angle_range, 180 + angle_range, 4))
    # writer = imageio.get_writer(args.filename_output, mode='I')
    training_images = []
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images, _, _ = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        images = pad(images, street_img, img_size)
        training_images.append(images)
    train_data = []
    for i in range(0, len(training_images), batch_size):
        train_data.append(torch.cat(training_images[i:i + batch_size], 0))
    
    return train_data


def pad(img, street_img, imgsize):
    # street_img = street_img.transpose(1, 2, 0)
    
    
    w, h = street_img.size
    if w == h:
        padded_img = street_img
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            padded_img.paste(street_img, (int(padding), 0))
    
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            padded_img.paste(street_img, (0, int(padding)))
    transform = transforms.Compose([transforms.Resize((imgsize, imgsize)),
                                    transforms.ToTensor()])

    street_img = transform(padded_img).unsqueeze(0)*255

    h, w = img.shape[2:]
    for i in range(h):
        for j in range(w):
            if img[0, 2, i, j]:
                street_img[:, :, 85 + i, 80 + j] = img[:, :, i, j]

    return street_img.squeeze(0)


# def pad(img, street_img, imgsize):
#     # street_img = street_img.transpose(1, 2, 0)
#
#     print(img.shape)
#     print(street_img.shape)
#     h, w = img.shape[2:]
#     for i in range(h):
#         for j in range(w):
#             if img[0, 2, i, j]:
#                 street_img[:, :, 70 + i, 110 + j] = img[:, :, i, j]
#     print(street_img.shape)
#     w, h = street_img.shape[2:]
#     if w == h:
#         padded_img = street_img
#     else:
#         # pdb.set_trace()
#         dim_to_pad = 1 if w < h else 2
#         if dim_to_pad == 1:
#             padding = (h - w) / 2
#             print(padding)
#             print(dim_to_pad)
#             # padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
#             padded = torch.full((1, 3, int(padding), h), 127)
#             print(padded.dtype)
#             print(street_img.dtype)
#             print(padded.shape)
#             padded_img = torch.cat([padded, street_img, padded], 2)
#
#         else:
#             padding = (w - h) / 2
#             # padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
#             # padded_img.paste(img, (0, int(padding)))
#             padded = torch.full((1, 3, w, int(padding)), 127)
#
#             padded_img = torch.cat([padded, street_img, padded], 3)
#
#     # resize = transforms.Resize((imgsize, imgsize))
#     # padded_img = F.interpolate(padded_img, (imgsize, imgsize))  # choose here
#     print(padded_img.shape)
#     return street_img.squeeze(0)


img = torch.from_numpy(imageio.imread(img_path).transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)
# street_img = torch.from_numpy(imageio.imread(street_img_path).transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)
street_img = Image.open(street_img_path).convert('RGB')
street_img = pad(img, street_img, 416)
street_img = street_img.data.numpy().transpose((1, 2, 0))
print(street_img.shape)
cv2.imwrite('/home/zhouge/Pictures/test_street.jpg', street_img)
