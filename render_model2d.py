#

import math
import os

# import neural_renderer as nr
import numpy as np
# torch.cuda.set_device(5)
# # os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
# torch.backends.cudnn.benchmark = True
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
import pdb
import imageio
from models import *
from utils.utils import *
from median_pool import MedianPool2d
# from adv_generator import Generator, weights_init_normal


# class RenderModel:
#     def __init__(self, config):
#         self.config = config
#         renderer = nr.Renderer(camera_mode='look_at')
#         renderer.perspective = False
#         renderer.light_intensity_directional = 0.0
#         renderer.light_intensity_ambient = 1.0
#         self.renderer = renderer
#
#     def render(self, vertices, faces, textures, angle):
#         # pdb.set_trace()
#         self.renderer.eye = nr.get_points_from_angles(self.config.d, self.config.e, angle)
#         images, _, _ = self.renderer(vertices, faces,
#                                      textures)  # [batch_size, RGB, image_size, image_size]
#         image = torch.flip(images, [3])
#
#         return image


def paste(img, p_image, h_pos, w_pos):
    
    '''

    :param img:
    :param p_image:
    :return: pasted img

    paste 2d img rasterized from mesh onto background imgs
    '''
    height, width = p_image.shape[2:]
    i_h, i_w = img.shape[2:]

    h_pad_len = height - i_h

    w_pad_len = width - i_w
    # paste

    h_pos = int(h_pos)
    w_pos = int(w_pos)
    h_bottom = h_pos
    h_top = h_pad_len - h_bottom
    w_top = w_pos
    w_bottom = w_pad_len - w_top
    # TODO:padiing img
    dim = (w_top, w_bottom, h_top, h_bottom)
    img = F.pad(img, dim, 'constant', value=0.)
    # print(img.shape)
    # print(p_image.shape)
    pasted_img = torch.where(img == 0., p_image, img)

    return pasted_img


# class AdvTrain(nn.Module):
#     def __init__(self, config, width, height):
#         super(AdvTrain, self).__init__()
#         self.config = config
#         self.model_width = width
#         self.model_height = height
#         self.min_contrast = 0.8
#         self.max_contrast = 1.2
#         self.min_brightness = -0.1
#         self.max_brightness = 0.1
#         self.noise_factor = 0.10
#         self.medianpooler = MedianPool2d(3, same=True)
#         self.Renderer = RenderModel(self.config)
#
#     def forward(self, universal_logo, vertices, faces, logo_index, target_image,
#                 bk_image, angle, conventional=False):
#         universal_logo = self.medianpooler(universal_logo.unsqueeze(0)).squeeze(0)
#         universal_logo = universal_logo.permute(1, 2, 0).contiguous().view(-1, 3)
#         target_image = target_image.cuda()
#         target_image = target_image.cuda()
#         vertices = vertices.cuda()
#         faces = faces.cuda()
#
#         contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).cuda()
#         brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness)
#         brightness = brightness.expand(universal_logo.shape).cuda()
#         noise = torch.FloatTensor(universal_logo.size()).uniform_(-1, 1) * self.noise_factor
#         universal_logo = universal_logo * contrast + brightness + noise.cuda()
#         universal_logo = torch.clamp(universal_logo, min=1e-7, max=0.9999999)
#         if self.config.consistent:
#             universal_logo = universal_logo.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2) \
#                 .expand(self.config.depth * self.config.width * self.config.height, 4, 4, 4, 3)
#         else:
#             universal_logo = universal_logo
#         if self.config.conventional:
#             universal_logo = universal_logo[: faces.shape[1]].unsqueeze(0).cuda()
#         else:
#             universal_logo = universal_logo[logo_index].unsqueeze(0).cuda()
#         adversarial_logo = universal_logo
#         # print(vertices.shape)
#         # print(faces.shape)
#         # print(adversarial_logo.shape)
#         # self.renderer.eye = nr.get_points_from_angles(self.config.d, self.config.e, angle)
#         logo_image = self.Renderer.render(vertices, faces,
#                                                 adversarial_logo, angle)  # [batch_size, RGB, image_size, image_size]
#         # tv = self.total_variation(logo_images[0])
#         # tv_loss = tv * 2.5
#
#         merge = torch.where(logo_image == 0., target_image, logo_image)
#         # clean_images = self.paste(target_image, bk_image).detach().cpu().numpy().transpose(0, 2, 3, 1)
#         # adv_images = self.paste(merge, bk_image).detach().cpu().numpy().transpose(0, 2, 3, 1)
#         # if self.config.paper_mtl:
#         #     self.paper_mtl(merge, logo_image,
#         #                    target_image, clean_images, adv_images, m_batch, angle)
#         training_images = self.augment(merge, bk_image)
#
#         training_images = F.interpolate(training_images, (self.model_height, self.model_width),
#                                         mode='bilinear')
#         # img = p_img_batch.detach().cpu().data[0, :, :, ]
#         # img = transforms.ToPILImage()(img)
#         # img.save('data/result{}.jpg'.format(m_batch))
#         # output = self.darknet_model(p_img_batch)
#         # dis_loss = self.prob_extractor(output)
#         # neg_count = self.calc_acc(output, self.darknet_model.num_classes, self.config.target)
#         return training_images, logo_image
#
#     def augment(self, img, bk_image, number=4):
#
#         # pdb.set_trace()
#         i_h, i_w = img.shape[2:]
#         size = [1.0]
#         aug_imgs = []
#         for scale in size:
#             rots = torch.linspace(-20, 20, number)
#             background = bk_image.clone()
#             poses = torch.linspace(self.model_width, 0, number + 2)
#             for pi, (rot, pos) in enumerate(zip(rots, poses[2:])):
#                 image = img.clone()
#                 image = F.interpolate(image, (int(scale * i_h), int(scale * i_w)), mode='bilinear')
#                 angle = float(rot) * math.pi / 180
#                 theta = torch.tensor([
#                     [math.cos(angle), math.sin(-angle), 0],
#                     [math.sin(angle), math.cos(angle), 0]
#                 ], dtype=torch.float).cuda()
#                 grid = F.affine_grid(theta.unsqueeze(0), image.size()).cuda()
#                 output = F.grid_sample(image, grid)
#                 output = output.expand(bk_image.shape[0], -1, -1, -1)
#                 background = paste(output, background, 0, pos)
#                 # print(background.device)
#             aug_imgs.append(background)
#         aug_imgs = torch.cat(aug_imgs, 0)
#
#         return aug_imgs


class AdvPatch(nn.Module):
    def __init__(self, config, width, height):
        super(AdvPatch, self).__init__()
        self.config = config
        self.model_width = width
        self.model_height = height
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.medianpooler = MedianPool2d(3, same=True)
    
    def forward(self, universal_logo,  contour_ref, target_image, bk_image):
        height, width = universal_logo.shape[1:]
        target_image = target_image.cuda()
        universal_logo = self.medianpooler(universal_logo.unsqueeze(0))
        # universal_logo = universal_logo.permute(1, 2, 0).contiguous().view(-1, 3)
        if contour_ref == 'G':
            contour = self.logo_G(self.config.width)
        elif contour_ref == 'H':
            contour = self.logo_h(self.config.width, self.config.height)
        contour = contour.unsqueeze(0).unsqueeze(0)
        contour = contour.expand(-1, 3, -1, -1).cuda()
        contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).cuda()
        brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.expand(universal_logo.shape).cuda()
        noise = torch.FloatTensor(universal_logo.shape).uniform_(-1, 1) * self.noise_factor
        universal_logo = universal_logo * contrast + brightness + noise.cuda()
        universal_logo = torch.clamp(universal_logo, min=1e-7, max=0.9999999)
        if contour_ref == 'H':
            scale = 0.35
        elif contour_ref == 'G':
            scale = 0.45
        logo_patch = torch.where(contour == 0, contour, universal_logo)
        logo_patch = F.interpolate(logo_patch, (int(height*scale),int(width*scale)),
                                        mode='bilinear')
        l_height, l_width = logo_patch.shape[2:]
        t_height, t_width = target_image.shape[2:]
        # print(logo_patch.shape)
        h_pos, w_pos = int((t_height - l_height)/2), int((t_width - l_width)/2)
        # print(h_pos, w_pos)
        merge = paste(logo_patch, target_image, h_pos, w_pos)
        # img = merge.detach().cpu().data[0, :, :, ]
        # img = transforms.ToPILImage()(img)
        # img.save('data/result{}.jpg'.format(0))
        training_images = self.augment(merge, bk_image)
        
        training_images = F.interpolate(training_images, (self.model_height, self.model_width),
                                        mode='bilinear')
        # img = p_img_batch.detach().cpu().data[0, :, :, ]
        # img = transforms.ToPILImage()(img)
        # img.save('data/result{}.jpg'.format(m_batch))
        # output = self.darknet_model(p_img_batch)
        # dis_loss = self.prob_extractor(output)
        # neg_count = self.calc_acc(output, self.darknet_model.num_classes, self.config.target)
        return training_images
    
    def augment(self, img, bk_image, number=4):
        
        # pdb.set_trace()
        i_h, i_w = img.shape[2:]
        size = [1.0]
        aug_imgs = []
        for scale in size:
            rots = torch.linspace(-20, 20, number)
            background = bk_image.clone()
            poses = torch.linspace(self.model_width, 0, number + 2)
            for pi, (rot, pos) in enumerate(zip(rots, poses[2:])):
                image = img.clone()
                image = F.interpolate(image, (int(scale * i_h), int(scale * i_w)), mode='bilinear')
                angle = float(rot) * math.pi / 180
                theta = torch.tensor([
                    [math.cos(angle), math.sin(-angle), 0],
                    [math.sin(angle), math.cos(angle), 0]
                ], dtype=torch.float).cuda()
                grid = F.affine_grid(theta.unsqueeze(0), image.size()).cuda()
                output = F.grid_sample(image, grid)
                output = output.expand(bk_image.shape[0], -1, -1, -1)
                background = paste(output, background, 0, pos)
                # print(background.device)
            aug_imgs.append(background)
        aug_imgs = torch.cat(aug_imgs, 0)
        
        return aug_imgs

    def logo_h(self, width, height):
        H = torch.zeros(width, height)
        x_pos1 = int(width * 0.4)
        x_pos2 = int(width * 0.6)
        y_pos1 = int(height * 0.4)
        y_pos2 = int(height * 0.6)
        H[:x_pos1, :] = 1
        H[x_pos2:, :] = 1
        H[:, y_pos1:y_pos2] = 1
        return H.t()

    def logo_G(self, length):
        G = torch.zeros(length, length)
        radius = (length - 1) / 2
    
        for i in range(length):
            for j in range(length):
                if radius * 0.6 < ((i - radius) ** 2 + (j - radius) ** 2) ** 0.5 < radius:
                    G[i][j] = 1
                if 0 < math.atan((i - radius) / (j - radius)) < math.pi / 4 and i > int(radius):
                    G[i][j] = 0
                if int(radius * 0.8) < i < int(radius * 1.2) and int(radius) < j:
                    G[i][j] = 1
                if ((i - radius) ** 2 + (j - radius) ** 2) ** 0.5 >= radius:
                    G[i][j] = 0
        return torch.flip(G, [0])
    

# class AdvTest(nn.Module):
#     def __init__(self, config, width, height):
#         super(AdvTest, self).__init__()
#         self.config = config
#         self.model_width = width
#         self.model_height = height
#
#
#     def forward(self, logo_image, target_image, bk_image):
#         merge = torch.where(logo_image == 0., target_image, logo_image)
#         merge = merge.expand(bk_image.shape[0], -1, -1, -1)
#         pos = int((self.model_width - merge.shape[-1])/2)
#         testing_images = paste(merge, bk_image, 0, pos)
#
#         testing_images = F.interpolate(testing_images, (self.model_height, self.model_width),
#                                        mode='bilinear')
#         return testing_images
    
def dis_loss_calc(output, num_classes,
                  num_anchors, target_id=0):
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)
    output = output.view(batch * num_anchors, 5 + num_classes, h * w)
    output = output.transpose(0, 1).contiguous()
    output = output.view(5 + num_classes, batch * num_anchors * h * w)
    det_confs = torch.sigmoid(output[4])
    cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1)))
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    cls_max_ids = torch.eq(cls_max_ids, target_id).float()
    det_human_conf = torch.where(cls_max_ids == 0., cls_max_ids, det_confs)
    det_human_conf = det_human_conf.contiguous().view(batch, -1)
    target_conf, target_conf_id = torch.max(det_human_conf, 1)
    return torch.mean(target_conf)
    
    
def calc_acc(output, num_classes,
             num_anchors, target_id):
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)
    output = output.view(batch * num_anchors, 5 + num_classes, h * w)
    output = output.transpose(0, 1).contiguous()
    output = output.view(5 + num_classes, batch * num_anchors * h * w)
    det_confs = torch.sigmoid(output[4])
    cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1)))
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    cls_max_ids = torch.eq(cls_max_ids, target_id).float()
    det_human_conf = torch.where(cls_max_ids == 0., cls_max_ids, det_confs)
    det_human_conf = det_human_conf.contiguous().view(batch, -1)
    target_conf, target_conf_id = torch.max(det_human_conf, 1)
    target_conf = target_conf.detach().cpu().data
    count = torch.sum(target_conf < 0.5).float().data
    return count
    
# def detect(image, darknet_model, num_classes, savename):
#     import cv2
#     use_cuda = 1
#     if num_classes == 20:
#         namesfile = 'data/voc.names'
#     elif num_classes == 80:
#         namesfile = 'data/coco.names'
#     else:
#         namesfile = 'data/names'
#     image = image * 255
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     boxes = do_detect(darknet_model, image, 0.5, 0.4, use_cuda)
#     class_names = load_class_names(namesfile)
#     plot_boxes_cv2(image, boxes, savename=savename, class_names=class_names)
        
# def paper_mtl(merge, logo_image,
#               target_image, clean_images, adv_images, m_batch, angle):
#     if angle in [170, 189, -10, 9]:
#         # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
#         syn_image = merge.detach().cpu().numpy()[0].transpose(1, 2, 0)
#         # raw_logo_image = raw_logo_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
#         logo_image = logo_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
#         target_image = target_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
#         detect(clean_images[0], 'data/pics/clean{}_{}.png'.format(m_batch, angle))
#         detect(adv_images[0], 'data/pics/adv{}_{}.png'.format(m_batch, angle))
#         imageio.imwrite('data/pics/target{}_{}.png'.format(m_batch, angle), 255 * target_image)
#         # imageio.imwrite('data/pics/raw_logo{}_{}.png'.format(m_batch, angle), 255 * raw_logo_image)
#         imageio.imwrite('data/pics/logo{}_{}.png'.format(m_batch, angle), 255 * logo_image)
#         imageio.imwrite('data/pics/syn{}_{}.png'.format(m_batch, angle), 255 * syn_image)
        
    
