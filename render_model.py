#

import neural_renderer as nr
# torch.cuda.set_device(5)
# # os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
# torch.backends.cudnn.benchmark = True
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from darknet import Darknet


class RenderModel(nn.Module):
    def __init__(self, config):
        super(RenderModel, self).__init__()
        self.config = config
        self.grad_textures = nn.Parameter(torch.full((99964, 4, 4, 4, 3), 0.5).cuda())
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()
        for p in self.darknet_model.parameters():
            p.requires_grad = False
        self.linear1 = nn.Linear(3, 24)
        self.linear2 = nn.Linear(12, 192)
        self.softmax = nn.Softmax(dim=1)
        self.convtranspose = nn.ConvTranspose3d(3, 3, (3, 3, 3), stride=1)
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer
    
    def forward(self, mesh, bk_image, angle):
        # self.renderer.eye = nr.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        
        # print(self.grad_indexs.shape)
        vertices = mesh['target_vertices'].cuda()
        faces = mesh['target_faces'].cuda()
        textures = mesh['target_textures'].cuda()
        logo_indexs = mesh['logo_indexs'].cuda()
        logo_scale = mesh['logo_scale'].cuda()
        # t_size = list(textures.size())
        # grad_indexs = []
        # grad_size = t_size.copy()
        # grad_size[0] = 1
        # for index in logo_indexs:
        #     grad_index = torch.full(tuple(grad_size), index, dtype=torch.long)
        #     grad_indexs.append(grad_index)
        # grad_indexs = torch.cat(grad_indexs, 0).cuda()
        # grad_textures = self.generator(logo_scale)
        # grad_t_size = t_size.copy()
        # grad_t_size[0] = len(logo_indexs)
        
        # textures = textures.scatter(0, grad_indexs, self.grad_textures).unsqueeze(0)
        # textures = textures.scatter(0, grad_indexs, grad_textures).unsqueeze(0)
        textures = self.grad_textures.unsqueeze(0)
        self.renderer.eye = nr.get_points_from_angles(2.0, 0., angle)
        image, _, _ = self.renderer(vertices, faces,
                                    textures)  # [batch_size, RGB, image_size, image_size]
        image = torch.flip(image, [-1])
        
        training_images = self.paste(image, bk_image)
        
        p_img_batch = F.interpolate(training_images, (self.darknet_model.height, self.darknet_model.width))
        img = p_img_batch.detach().cpu().data[0, :, :, ]
        img = transforms.ToPILImage()(img)
        img.save('data/result.jpg')
        output = self.darknet_model(p_img_batch)
        
        # del p_img_batch, img, training_images, image, textures, vertices, faces
        # torch.cuda.empty_cache()
        # ref_images = torch.randn(training_images.shape).cuda()
        # loss = torch.sum((training_images - ref_images) ** 2)
        return output
    
    def paste(self, img, bk_image):
        
        '''
        
        :param img:
        :param bk_image:
        :return: pasted img
        
        paste 2d img rasterized from mesh onto background imgs
        '''
        i_h, i_w = img.shape[2:]
        # rescale
        scale = 0.75
        
        img = F.interpolate(img, size=[int(scale * i_h), int(scale * i_w)], mode='bilinear')
        img = img
        i_h, i_w = img.shape[2:]
        
        h_pad_len = self.darknet_model.height - i_h
        
        w_pad_len = self.darknet_model.height - i_w
        paste_imgs = []
        # paste
        for i in range(len(img)):
            # h_pos = rd.randint(0, h_pad_len)
            h_pos = int(h_pad_len * 0.75)
            
            w_pos = int(w_pad_len * 0.5)
            # w_pos = rd.randint(0, w_pad_len)
            
            h_top = h_pos
            h_bottom = h_pad_len - h_top
            w_top = w_pos
            w_bottom = w_pad_len - w_top
            # h_top = int((img_size - i_h) / 2) if (img_size - i_h) % 2 == 0 else int((img_size - i_h) / 2) + 1
            # h_bottom = int((img_size - i_h) / 2)
            # w_top = int((img_size - i_w) / 2) if (img_size - i_h) % 2 == 0 else int((img_size - i_h) / 2) + 1
            # w_bottom = int((img_size - i_w) / 2)
            # TODO:padiing img
            dim = (w_top, w_bottom, h_top, h_bottom)
            img = F.pad(img, dim, 'constant', value=0.)
            paste_imgs.append(img)
        
        pasted_imgs = torch.cat(paste_imgs, 0)
        
        pasted_len = len(pasted_imgs)
        pasted_imgs = pasted_imgs.repeat(len(bk_image), 1, 1, 1)
        # print(augmented_img.shape)
        bk_image = bk_image.repeat(pasted_len, 1, 1, 1)
        # print(bk_image)
        
        pasted_img = torch.where(pasted_imgs == 0., bk_image, pasted_imgs)
        
        return pasted_img
    
    def texture_augment(self, vertices, faces, textures):
        '''
        
        :param vertices:
        :param faces:
        :param textures:
        :return: augmented params
        
        
        augment mesh textures
        '''
        augment_element = ((20000, 30000, 0),
                           (50000, 60000, 2),
                           (70000, 80000, 1),
                           (40000, 5500, 0))
        vertices = vertices.repeat(len(augment_element), 1, 1, 1, 1, 1)
        faces = faces.repeat(len(augment_element), 1, 1, 1, 1, 1)
        textures = textures.repeat(len(augment_element), 1, 1, 1, 1, 1)
        for aug_i, aug in augment_element:
            textures[aug_i, aug[0]:aug[1], :, :, :, aug[2]] = 0.
        return vertices, faces, textures
    
    def self_atten(self, inputs):
        '''
        
        :param inputs:
        :return: self attention
        '''
        attn = inputs.matmul(inputs.transpose(0, 1))
        attn = self.softmax(attn)
        inputs = attn.matmul(inputs)
        return inputs
    
    def generator(self, inputs):
        '''
        
        :param inputs:
        :return: generated logo textures for gradient descend
        '''
        inputs_shape = list(inputs.shape)
        
        # self attention
        # print(inputs.shape)
        inputs = self.self_atten(inputs)
        
        # linear generator
        # inputs = self.linear1(inputs)
        # inputs = self.linear2(inputs)
        
        # conv transpose generator
        inputs = self.linear1(inputs)
        # print(inputs.shape)
        inputs = inputs.view((inputs_shape[0], 3, 2, 2, 2))
        inputs = self.convtranspose(inputs)
        # print(inputs.shape)
        # return inputs.unsqueeze(0).permute((0, 1, 3, 4, 5, 2))
        return inputs.permute((0, 2, 3, 4, 1))

# import tqdm

#
# from load_data import *


# input = torch.randn((12, 3, 4, 4, 4))
# conv3d = nn.Conv3d(3, 3, (3, 3, 3), stride=1)
# transconv3d = nn.ConvTranspose3d(3, 3, (3, 3, 3), stride=1, padding=0)
# # embedding = nn.Linear(3, 64)
# # self-attention
# # softmax = nn.Softmax(dim=1)
# # attn = input.matmul(input.transpose(0, 1))
# # input = attn.matmul(input)
# output = conv3d(input)
# print(output.shape)
# output = transconv3d(output)
# # output = embedding(input)
# print(output.shape)
# # print(output)

# class Model(nn.Module):
#     def __init__(self, filename_obj, filename_ref, filename_logo, img_size):
#         super(Model, self).__init__()
#         vertices, faces, textures = nr.load_obj(filename_obj, load_texture=True)
#         self.register_buffer('vertices', vertices[None, :, :])
#         self.register_buffer('faces', faces[None, :, :])
#         t_size = list(textures.size())
#         self.register_buffer('textures', textures.requires_grad_(False))
#         # load reference image
#         with open(filename_logo, 'rb') as logo_file:
#             logo_indexs = np.array(pickle.load(logo_file))
#
#         grad_t_size = t_size.copy()
#         grad_t_size[0] = len(logo_indexs)
#         self.grad_textures = nn.Parameter(torch.full(grad_t_size, 0.5).cuda())
#         # self.grad_textures = nn.Parameter(torch.randn(grad_t_size).cuda())
#
#         grad_indexs = []
#         grad_size = t_size.copy()
#         grad_size[0] = 1
#         for index in logo_indexs:
#             grad_index = torch.full(tuple(grad_size), index, dtype=torch.long)
#             grad_indexs.append(grad_index)
#         self.register_buffer('grad_indexs', torch.cat(grad_indexs, 0).cuda())
#         # self.register_buffer('grad_indexs', torch.from_numpy(logo_indexs))
#         # textures = textures.scatter_(0, grad_indexs, self.grad_textures)
#
#         image_ref = Image.open(filename_ref).convert('RGB')
#         self.register_buffer('image_ref', self.pad(image_ref, img_size))
#
#         # setup renderer
#         renderer = nr.Renderer(camera_mode='look_at')
#         renderer.perspective = False
#         renderer.light_intensity_directional = 0.0
#         renderer.light_intensity_ambient = 1.0
#         self.renderer = renderer
#
#     def forward(self, img_size, batch_size, i_batch, angle):
#         # self.renderer.eye = nr.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
#         # pdb.set_trace()
#         # print(self.grad_indexs.shape)
#
#         textures = self.textures.scatter(0, self.grad_indexs, self.grad_textures).unsqueeze(0)
#         # textures = self.textures.unsqueeze(0)
#         # textures[:, self.grad_indexs, :, :, :, :] = self.grad_textures.unsqueeze(0)
#         # textures = textures.unsqueeze(0)
#         # print(textures.size())
#         # start = 172 + i_batch * angle_range
#         # end = start + angle_range
#         loop = tqdm(range(batch_size))
#         training_images = []
#         # ref_images = []
#         self.renderer.eye = nr.get_points_from_angles(2.0, 0., angle)
#         image, _, _ = self.renderer(self.vertices, self.faces,
#                                     textures)  # [batch_size, RGB, image_size, image_size]
#         image = torch.flip(image, [-1])
#
#         for num_i, num in enumerate(loop):
#             loop.set_description('Padding')
#             # self.renderer.eye = nr.get_points_from_angles(2.0, 0., azimuth)
#             # image, _, _ = self.renderer(self.vertices, self.faces,
#             #                             textures)  # [batch_size, RGB, image_size, image_size]
#             # image = torch.flip(image, [-1])
#             training_image = self.paste(image, img_size, i_batch, num / batch_size)
#             training_images.append(training_image)
#             # ref_images.append(ref_image)
#         print(torch.cuda.memory_allocated())
#         training_images = torch.cat(training_images, 0)
#         del image, textures
#         torch.cuda.empty_cache()
#         # ref_images = torch.randn(training_images.shape).cuda()
#         # loss = torch.sum((training_images - ref_images) ** 2)
#         return training_images
#
#     def paste(self, img, img_size, i_batch, num):
#         # pdb.set_trace()
#         # pdb.set_trace()
#         # print(img.size())
#         i_h, i_w = img.shape[2:]
#         # print(i_h, i_w)
#         # scale = rd.uniform(0.5, 1)
#         scale = 0.75
#         # print(scale)
#         img = F.interpolate(img, size=[int(scale * i_h), int(scale * i_w)], mode='bilinear')
#         img = img.squeeze(0)
#         i_h, i_w = img.shape[1:]
#         h_pad_len = img_size - i_h
#         # h_pos = rd.randint(0, h_pad_len)
#         h_pos = int(h_pad_len * i_batch)
#
#         w_pad_len = img_size - i_w
#         # w_pos = rd.randint(0, w_pad_len)
#         w_pos = int(w_pad_len * num)
#
#         h_top = h_pos
#         h_bottom = h_pad_len - h_top
#         w_top = w_pos
#         w_bottom = w_pad_len - w_top
#         # h_top = int((img_size - i_h) / 2) if (img_size - i_h) % 2 == 0 else int((img_size - i_h) / 2) + 1
#         # h_bottom = int((img_size - i_h) / 2)
#         # w_top = int((img_size - i_w) / 2) if (img_size - i_h) % 2 == 0 else int((img_size - i_h) / 2) + 1
#         # w_bottom = int((img_size - i_w) / 2)
#         # TODO:padiing img
#         dim = (w_top, w_bottom, h_top, h_bottom)
#         img = F.pad(img, dim, 'constant', value=0.)
#
#         pasted_img = torch.where(img == 0., self.image_ref, img)
#
#         return pasted_img.unsqueeze(0)
#
#     def pad(self, image_ref, img_size):
#         # w, h = image_ref.size
#         # if w == h:
#         #     padded_img = image_ref
#         # else:
#         #     dim_to_pad = 1 if w < h else 2
#         #     if dim_to_pad == 1:
#         #         padding = (h - w) / 2
#         #         padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
#         #         padded_img.paste(image_ref, (int(padding), 0))
#         #
#         #     else:
#         #         padding = (w - h) / 2
#         #         padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
#         #         padded_img.paste(image_ref, (0, int(padding)))
#
#         transform = transforms.Compose([transforms.Resize((img_size, img_size)),
#                                         transforms.ToTensor()])
#
#         # padded_img = transform(padded_img).cuda()
#         padded_img = transform(image_ref).cuda()
#
#         return padded_img
