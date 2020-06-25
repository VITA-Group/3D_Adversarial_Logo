#

import math
import os
import cv2 as cv
import neural_renderer as nr
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
from darknet import Darknet
import imageio
from utils import *
from median_pool import MedianPool2d
# from adv_generator import Generator, weights_init_normal

class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = lambda obj, cls: obj * cls

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls ) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls , :]  # [batch, 80, 1805]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        confs_if_object = output_objectness #confs_for_class * output_objectness
        confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.loss_target(output_objectness, confs_for_class)
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        return torch.mean(max_conf)

class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """
    
    def __init__(self):
        super(TotalVariation, self).__init__()
    
    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """
    
    def __init__(self, printability_file, img_size):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, img_size),
                                               requires_grad=False)
    
    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array.cuda() + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)
    
    def get_printability_array(self, printability_file, side):
        printability_list = []
        
        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))
        
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)
        
        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class RenderModel(nn.Module):
    def __init__(self, config):
        super(RenderModel, self).__init__()
        self.config = config
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        # if self.config.cuda is not '-1':
        #     torch.cuda.set_device(self.config.cuda)
        #     self.device = torch.device('cuda')
        # else:
        #     self.device = torch.device('cpu')
        # if self.config.consistent:
        #     self.grad_textures = grad_textutres.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)\
        #         .expand(self.config.depth * self.config.width * self.config.height, 4, 4, 4, 3)
        # else:
        #     self.grad_textures = grad_textutres
        # self.grad_textures = grad_textutres
        # self.grad_textures = grad_textutres.expand(self.config.depth * self.config.width * self.config.height,
        #                                            4, 4, 4, 3)
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()
        # for p in self.darknet_model.parameters():
        #     p.requires_grad = False
        # self.cubic = nn.Parameter(torch.full((1, 4, 4, 4, 3), 0.5).cuda())
        # self.Xembedding = torch.nn.Embedding(100, 256)
        # self.Yembedding = torch.nn.Embedding(100, 256)
        # self.Zembedding = torch.nn.Embedding(100, 256)
        # self.linear1 = nn.Linear(768, 192)
        # self.linear2 = nn.Linear(256, 64)
        # self.linear3 = nn.Linear(256, 64)
        # self.linear4 = nn.Linear(192, 192)
        # self.softmax = nn.Softmax(dim=2)
        # self.convtranspose = nn.ConvTranspose3d(3, 3, (3, 3, 3), stride=1)
        self.prob_extractor = MaxProbExtractor(0, 80).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.image_size).cuda()
        self.total_variation = TotalVariation().cuda()
        self.medianpooler = MedianPool2d(7, same=True)
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer

    def forward(self, universal_logo_raw, vertices, faces, logo_index, target,
                bk_image, angle, i_batch, m_batch, train_patch=False, conventional=False):
        universal_logo_raw = self.medianpooler(universal_logo_raw.unsqueeze(0)).squeeze(0)
        self.renderer.eye = nr.get_points_from_angles(self.config.d, self.config.e, angle)
    
        target_vertices, target_faces, target_textures = target
        target_vertices = target_vertices.cuda()
        target_faces = target_faces.cuda()
        target_textures = target_textures.cuda()
    
        target_images, _, _ = self.renderer(target_vertices, target_faces,
                                            target_textures)
        target_image = torch.flip(target_images, [3])
    
        if train_patch:
            height, width = universal_logo_raw.shape[1:]
            if self.config.logo_ref == 'G':
                contour = self.logo_G(self.config.width)
                scale = 0.45
            elif self.config.logo_ref == 'H':
                contour = self.logo_h(self.config.width, self.config.height)
                scale = 0.35
            contour = contour.unsqueeze(0).unsqueeze(0)
            contour = contour.expand(-1, 3, -1, -1).cuda()
            # print(contour.shape)
            contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).cuda()
            brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness)
            brightness = brightness.expand(universal_logo_raw.shape).cuda()
            noise = torch.FloatTensor(universal_logo_raw.shape).uniform_(-1, 1) * self.noise_factor
            universal_logo = universal_logo_raw * contrast + brightness + noise.cuda()
            universal_logo = torch.clamp(universal_logo, min=1e-6, max=0.999999)
            logo_patch = torch.where(contour == 0, contour, universal_logo.unsqueeze(0))
            logo_patch = F.interpolate(logo_patch, (int(height * scale), int(width * scale)),
                                       mode='bilinear')
            l_height, l_width = logo_patch.shape[2:]
            t_height, t_width = target_image.shape[2:]
            # print(logo_patch.shape)
            h_pos, w_pos = 120, int((t_width - l_width) / 2)
            # print(h_pos, w_pos)
            logo_image = self.pad_logo(logo_patch, h_pos, w_pos)
        else:
        
            universal_logo_raw = universal_logo_raw.permute(1, 2, 0).contiguous().view(-1, 3)
            vertices = vertices.cuda()
            faces = faces.cuda()
            # logo_index = self.index_revise(logo_scale)
            # pdb.set_trace()
            # print(torch.max(logo_index))
            # print(logo_index)
        
            # universal_logo = self.medianpooler(universal_logo.unsqueeze(0))
            contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).cuda()
            brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness)
            brightness = brightness.expand(universal_logo_raw.shape).cuda()
            noise = torch.FloatTensor(universal_logo_raw.size()).uniform_(-1, 1) * self.noise_factor
            universal_logo = universal_logo_raw * contrast + brightness + noise.cuda()
            universal_logo = torch.clamp(universal_logo, min=1e-7, max=0.9999999)
            if self.config.consistent:
                universal_logo = universal_logo.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2) \
                    .expand(self.config.depth * self.config.width * self.config.height, 4, 4, 4, 3)
                # universal_logo_raw = universal_logo_raw.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2) \
                #     .expand(self.config.depth * self.config.width * self.config.height, 4, 4, 4, 3)
            else:
                universal_logo = universal_logo
                # universal_logo_raw = universal_logo_raw
            if self.config.conventional:
                universal_logo = universal_logo[: faces.shape[1]].unsqueeze(0).cuda()
                # universal_logo_raw = universal_logo_raw[: faces.shape[1]].unsqueeze(0).cuda()
            else:
                # print('using multiple version')
                universal_logo = universal_logo[logo_index].unsqueeze(0).cuda()
                # universal_logo_raw = universal_logo_raw[logo_index].unsqueeze(0).cuda()
            # print(grad_textures)
            # contour_textures = torch.full(universal_logo.shape, 0.5).cuda()
        
            # print(grad_textures.shape)
            adversarial_logo = universal_logo
            # adversarial_logo = torch.clamp(universal_logo, min=1e-7, max=0.999999)
        
            # target_images, _, _ = self.renderer(target_vertices, target_faces,
            #                                     target_textures)  # [batch_size, RGB, image_size, image_size]
            # target_image = torch.flip(target_images, [3])
        
            # print(vertices.device)
            # print(faces.device)
            # print(self.grad_textures.device)
            # print(grad_textures.device)
            logo_images, _, _ = self.renderer(vertices, faces,
                                              adversarial_logo)  # [batch_size, RGB, image_size, image_size]
            # raw_logo_images, _, _ = self.renderer(vertices, faces,
            #                                       universal_logo_raw)
            logo_image = torch.flip(logo_images, [3])
        # nps_loss = self.nps_calculator(self.pad_logo(universal_logo_raw))
        # tv_loss = self.total_variation(universal_logo_raw.contiguous().view(1, 3, self.config.height, self.config.width))
        # nps = self.nps_calculator(logo_images[0])
        tv = self.total_variation(logo_image[0])
        # nps_loss = nps * 0.01
        tv_loss = tv * 2.5
        # noise, contrast, brightness
        # contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).cuda()
        # brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness)
        # brightness = brightness.expand(raw_logo_images.shape).cuda()
        # noise = torch.FloatTensor(raw_logo_images.size()).uniform_(-1, 1) * self.noise_factor
        # logo_images = raw_logo_images * contrast + brightness + noise.cuda()
        # logo_images = torch.clamp(logo_images, min=1e-7, max=0.999999)
    
        # print(logo_images.device)
        # print(self.darknet_model.device)
        # logo_images = torch.clamp(logo_images, min=0.0, max=0.9999)
    
        merge = torch.where(logo_image == 0., target_image, logo_image)
    
        # image, _, _ = self.renderer(vertices, faces,
        #                             textures)  # [batch_size, RGB, image_size, image_size]
        # image = torch.flip(merge, [3])
    
        # clean_images = self.paste(target_image, bk_image).detach().cpu().numpy().transpose(0, 2, 3, 1)
        # adv_images = self.paste(merge, bk_image).detach().cpu().numpy().transpose(0, 2, 3, 1)
        # cur_dir = os.path.dirname(__file__)
        # mtl_filepath = os.path.join(cur_dir, 'data/image/material{}_angle{}.pkl'.format(m_batch, angle))
        # self.detect(adv_images[0], 'data/pics/clean{}_{}.png'.format(m_batch, angle))
        cur_dir = os.path.dirname(__file__)
        mtl_filepath = os.path.join(cur_dir, 'data/image_{}/material{}_angle{}.pkl'.format(self.config.logo_ref, m_batch, angle))
        w_edge = int(self.darknet_model.height - self.config.image_size)
        results = self.paste(merge, bk_image, 0, 0)
        results = self.paste(target_image, results, 0, w_edge)
        # if angle in [180, 0]:
            # self.paper_mtl(merge, logo_image,
            #                target_image, clean_images, adv_images, m_batch, angle)
            # imageio.imwrite('data/pics/clean{}_{}.png'.format(m_batch, angle), 255 * clean_images[0])
            # imageio.imwrite('data/pics/adv{}_{}.png'.format(m_batch, angle), 255 * adv_images[0])
            # torch.save({'mtl': [target_image.detach().cpu().data, merge.detach().cpu().data],
            #             'image': results.detach().cpu().data}, mtl_filepath)
        # preserve the images for paper
        # if self.config.paper_mtl:
            # self.paper_mtl(merge, logo_image,
            #                target_image, clean_images, adv_images, m_batch, angle)
            # imageio.imwrite('data/pics/clean{}_{}.png'.format(m_batch, angle), 255 * clean_images[0])
            # imageio.imwrite('data/pics/adv{}_{}.png'.format(m_batch, angle), 255 * adv_images[0])
            # torch.save({'mtl': [target_image.detach().cpu(), nerge.detach().cpu()],
            #             'image': [clean_images[0], adv_images[0]]}, mtl_filepath)
    
        # pdb.set_trace()
        # print(image.device)
        # print(bk_image.device)
        w_pos = np.random.randint(int(self.darknet_model.height - self.config.image_size))
        training_images = self.paste(merge, bk_image, 0, w_pos)
    
        p_img_batch = F.interpolate(training_images, (self.darknet_model.height, self.darknet_model.width),
                                    mode='bilinear')
    
        # img_filepath = os.path.join(cur_dir, 'data/image/image_angle{}_mesh{}.pkl'.format(angle, m_batch))
        # if angle in [180, 0, 177, 187, 9] and i_batch == 0:
        #     torch.save([clean_images[1], adv_images[1]], img_filepath)
        # img = p_img_batch.detach().cpu().data[0, :, :, ].numpy()
        # img = img.transpose(1, 2, 0)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # cv.imwrite('data/image{}/result{}.png'.format(self.config.logo_ref, m_batch), img * 255)
        output = self.darknet_model(p_img_batch)
        # dis_loss = self.dis_loss(output, self.darknet_model.num_classes, self.darknet_model.anchors,
        #                                     self.darknet_model.num_anchors, target_id=self.config.target)
        # dis_loss = self.prob_extractor(output)
        dis_loss = self.dis_loss(output, self.darknet_model.num_classes, self.darknet_model.anchors,
                                 self.darknet_model.num_anchors, 0)
        neg_count = self.calc_acc(output, self.darknet_model.num_classes, self.config.target)
        # del p_img_batch, img, training_images, image, textures, vertices, faces
        # torch.cuda.empty_cache()
        # ref_images = torch.randn(training_images.shape).cuda()
        # loss = torch.sum((training_images - ref_images) ** 2)
        return dis_loss, tv_loss, neg_count
    
    def pad(self, img, pos):
        '''

        :param img:
        :param bk_image:
        :return: pasted img

        paste 2d img rasterized from mesh onto background imgs
        '''
        i_h, i_w = img.shape[2:]
        
        h_pad_len = self.darknet_model.height - i_h
        
        w_pad_len = self.darknet_model.width - i_w
        # paste_imgs = []
        # paste
        # h_pos = rd.randint(0, h_pad_len)
        # h_pos = int(h_pad_len * 0.75)
        
        # w_pos = int(w_pad_len * 0.5)
        # w_pos = rd.randint(0, w_pad_len)
        
        h_top = h_pad_len
        h_bottom = 0
        w_top = int(pos)
        w_bottom = w_pad_len - w_top
        # h_top = int((img_size - i_h) / 2) if (img_size - i_h) % 2 == 0 else int((img_size - i_h) / 2) + 1
        # h_bottom = int((img_size - i_h) / 2)
        # w_top = int((img_size - i_w) / 2) if (img_size - i_h) % 2 == 0 else int((img_size - i_h) / 2) + 1
        # w_bottom = int((img_size - i_w) / 2)
        # TODO:padiing img
        dim = (w_top, w_bottom, h_top, h_bottom)
        img = F.pad(img, dim, 'constant', value=0.)
        
        return img

    def pad_logo(self, img, posy, posx):
        '''

        :param img:
        :param bk_image:
        :return: pasted img

        paste 2d img rasterized from mesh onto background imgs
        '''
        i_h, i_w = img.shape[2:]
    
        h_pad_len = self.config.image_size - i_h
    
        w_pad_len = self.config.image_size - i_w
        h_top = int(h_pad_len - posy)
        h_bottom = posy
        w_top = posx
        w_bottom = int(w_pad_len - w_top)
        dim = (w_top, w_bottom, h_top, h_bottom)
        img = F.pad(img, dim, 'constant', value=0.)
    
        return img
    
    def augment(self, img, bk_image, number=4):
        
        # pdb.set_trace()
        img = img
        i_h, i_w = img.shape[2:]
        size = [1.0]
        aug_imgs = []
        for scale in size:
            rots = torch.linspace(-20, 20, number)
            # print(rots)
            
            poses = torch.linspace(self.darknet_model.width, 0, number + 2)
            # print(poses)
            board = torch.zeros(1, 3, self.darknet_model.height, self.darknet_model.width).cuda()
            for pi, (rot, pos) in enumerate(zip(rots, poses[2:])):
                # print(poses[2:])
                # print(rot, pos)
                image = img.clone()
                # pallete = torch.zeros(image.shape).cuda()
                # channel = pi % 3
                # pallete[:, channel, int(i_h * 0.6):, :] = 1.
                # color_aug = torch.where(image != 0., pallete, image)
                # image = torch.where(color_aug != 0., color_aug, image)
                image = F.interpolate(image, (int(scale * i_h), int(scale * i_w)), mode='bilinear')
                angle = float(rot) * math.pi / 180
                theta = torch.tensor([
                    [math.cos(angle), math.sin(-angle), 0],
                    [math.sin(angle), math.cos(angle), 0]
                ], dtype=torch.float).cuda()
                # blank = torch.ones(target.shape)
                grid = F.affine_grid(theta.unsqueeze(0), image.size()).cuda()
                output = F.grid_sample(image, grid)
                output = self.pad(output, pos)
                board = torch.where(output == 0., board, output)
            # plt.imshow(board[0].numpy().transpose(1, 2, 0))
            # plt.savefig('/home/zhouge/Documents/aug{}.pdf'.format(scale), bbox_inches='tight')
            aug_imgs.append(board)
            aug_imgs = torch.cat(aug_imgs, 0)
            aug_len = len(aug_imgs)
            aug_imgs = aug_imgs.contiguous().repeat(len(bk_image), 1, 1, 1)
            bk_image = bk_image.contiguous().repeat(aug_len, 1, 1, 1)

            aug_imgs = torch.where(aug_imgs == 0., bk_image, aug_imgs)
            return aug_imgs
    
    def self_atten(self, inputs):
        '''

        :param inputs:
        :return: self attention
        '''
        # print(inputs.transpose(1,2).shape)
        attn = torch.bmm(inputs, inputs.transpose(1, 2))
        attn = self.softmax(attn)
        inputs = torch.bmm(attn, inputs)
        return inputs
    
    # def genrator(self, input):

    def dis_loss(self, output, num_classes, anchors, num_anchors, target_id=0, only_objectness=1,
                 validation=False):
        # anchor_step = len(anchors)/num_anchors
        anchor_step = len(anchors) // num_anchors
        if output.dim() == 3:
            output = output.unsqueeze(0)
        batch = output.size(0)
        assert (output.size(1) == (5 + num_classes) * num_anchors)
        h = output.size(2)
        w = output.size(3)
        # print(output.size())
        output = output.view(batch * num_anchors, 5 + num_classes, h * w)
        # print(output.size())
        output = output.transpose(0, 1).contiguous()
        # print(output.size())
        output = output.view(5 + num_classes, batch * num_anchors * h * w)
        # print(output.size())
    
        all_target_acc = []
        det_confs = torch.sigmoid(output[4])
        # print(det_confs.shape)
        cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1)))
        # print(cls_confs.size())
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
        cls_max_confs = cls_max_confs.view(-1)
        # print(cls_max_ids.shape)
        cls_max_ids = cls_max_ids.view(-1)
        # print(cls_max_ids.shape)
    
        # pdb.set_trace()
        # print(cls_max_ids[302])
        cls_max_ids = torch.eq(cls_max_ids, target_id).float()
        # print(cls_max_ids[302])
    
        det_human_conf = torch.where(cls_max_ids == 0., cls_max_ids, det_confs)
        # print(det_human_conf[302])
        det_human_conf = det_human_conf.contiguous().view(batch, -1)
    
        # print(det_human_conf.shape)
        target_conf, target_conf_id = torch.max(det_human_conf, 1)
        # print(target_conf_id)
        # print(target_conf)
        # print(cls_max_confs[302])
        # target_conf_acc = cls_max_confs.contiguous().view(batch, -1)
        # for ii, i in enumerate(target_conf_id):
        #     all_target_acc.append(target_conf_acc[ii][i].detach().cpu().data)
    
        # print(target_conf_acc)
        # print(target_conf)
        # print('loss_acc:', all_target_acc)
        # return torch.mean(target_conf), target_conf.data
        # print('loss_acc:', torch.stack(all_target_acc))
        # print('target_conf:', target_conf.detach().cpu().data)
        return torch.mean(target_conf)

    def paste(self, img, p_image, h_pos, w_pos):
    
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

    def generator(self, inputs):
        '''
        :param inputs:
        :return: generated logo textures for gradient descend
        '''
        # inputs /= 100
        inputs = inputs.long()
        batch_size = list(inputs.shape)[0]
        x, y, z = inputs.split(1, 1)
        x_embed = self.Xembedding(x)
        # x_embed = self.linear1(x_embed)
        y_embed = self.Yembedding(y)
        # y_embed = self.linear2(y_embed)
        z_embed = self.Zembedding(z)
        # z_embed = self.linear3(z_embed)
        # pdb.set_trace()
        # self attention
        
        # print(inputs.shape)
        embed = torch.cat([x_embed, y_embed, z_embed], 1)
        # print(embed.shape)

        # inputs = self.embedding(inputs) / 100
        # print(inputs.shape)
        inputs = self.self_atten(embed).contiguous().view(batch_size, 768)
        inputs = F.relu(self.linear1(inputs))
        # inputs = (embed + inputs)
        # linear generator
        # inputs = self.linear1(inputs)
        # inputs = self.linear2(inputs)

        # conv transpose generator
        # inputs = self.linear1(inputs)
        # inputs = F.relu(inputs)
        # print(inputs.shape)
        # inputs = inputs
        # inputs = torch.tanh(self.linear4(inputs)).contiguous().view((batch_size, 3, 4, 4, 4))
        inputs = torch.tanh(self.linear4(inputs).contiguous().view(batch_size, 3, 4, 4, 4))
        # print(inputs.shape)
        # return inputs.unsqueeze(0).permute((0, 1, 3, 4, 5, 2))
        return inputs.permute((0, 2, 3, 4, 1)).unsqueeze(0)

    def calc_acc(self, output, num_classes, target_id):
        
        # anchor_step = len(anchors) // num_anchors
        if output.dim() == 3:
            output = output.unsqueeze(0)
        batch = output.size(0)
        assert (output.size(1) == (5 + num_classes) * self.darknet_model.num_anchors)
        h = output.size(2)
        w = output.size(3)
        # print(output.size())
        output = output.view(batch * self.darknet_model.num_anchors, 5 + num_classes, h * w)
        # print(output.size())
        output = output.transpose(0, 1).contiguous()
        # print(output.size())
        output = output.view(5 + num_classes, batch * self.darknet_model.num_anchors * h * w)
        # print(output.size())
    
        all_target_acc = []
        det_confs = torch.sigmoid(output[4])
        # print(det_confs.shape)
        cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1)))
        # print(cls_confs.size())
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
        cls_max_confs = cls_max_confs.view(-1)
        # print(cls_max_ids.shape)
        cls_max_ids = cls_max_ids.view(-1)
        # print(cls_max_ids.shape)
    
        # pdb.set_trace()
        # print(cls_max_ids[302])
        cls_max_ids = torch.eq(cls_max_ids, target_id).float()
        # print(cls_max_ids[302])
    
        det_human_conf = torch.where(cls_max_ids == 0., cls_max_ids, det_confs)
        # print(det_human_conf[302])
        det_human_conf = det_human_conf.contiguous().view(batch, -1)
    
        # print(det_human_conf.shape)
        target_conf, target_conf_id = torch.max(det_human_conf, 1)
        # print(target_conf_id)
        # print(target_conf)
        # print(cls_max_confs[302])
        # target_conf_acc = cls_max_confs.contiguous().view(batch, -1)
        # for ii, i in enumerate(target_conf_id):
        #     all_target_acc.append(target_conf_acc[ii][i].detach().cpu().data)
    
        # print(target_conf_acc)
        # print(target_conf)
        # print('loss_acc:', all_target_acc)
        # return torch.mean(target_conf), target_conf.data
        # print('loss_acc:', torch.stack(all_target_acc))
        # print('target_conf:', target_conf.detach().cpu().data)
        # target_conf, target_conf_id = torch.max(det_human_conf, 1)
        target_conf = target_conf.detach().cpu().data
        count = torch.sum(target_conf < 0.6).float().data
        return count

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
    
    def detect(self, image, savename):
        import cv2
        use_cuda = 1
        num_classes = self.darknet_model.num_classes
        if num_classes == 20:
            namesfile = 'data/voc.names'
        elif num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'
        image = image * 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = do_detect(self.darknet_model, image, 0.5, 0.4, use_cuda)
        class_names = load_class_names(namesfile)
        plot_boxes_cv2(image, boxes, savename=savename, class_names=class_names)
        
    def paper_mtl(self, merge, logo_image,
                  target_image, clean_images, adv_images, m_batch, angle):
        if angle in [170, 189, -10, 9]:
            # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
            syn_image = merge.detach().cpu().numpy()[0].transpose(1, 2, 0)
            # raw_logo_image = raw_logo_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
            logo_image = logo_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
            target_image = target_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
            self.detect(clean_images[0], 'data/pics/clean{}_{}.png'.format(m_batch, angle))
            self.detect(adv_images[0], 'data/pics/adv{}_{}.png'.format(m_batch, angle))
            imageio.imwrite('data/pics/target{}_{}.png'.format(m_batch, angle), 255 * target_image)
            # imageio.imwrite('data/pics/raw_logo{}_{}.png'.format(m_batch, angle), 255 * raw_logo_image)
            imageio.imwrite('data/pics/logo{}_{}.png'.format(m_batch, angle), 255 * logo_image)
            imageio.imwrite('data/pics/syn{}_{}.png'.format(m_batch, angle), 255 * syn_image)
        
    # def index_revise(self, scale):
    #
    #     list_index = []
    #     if self.config.depth == 1:
    #         depth = 0
    #     else:
    #         depth = self.config.side_length**2
    #     # pdb.set_trace()
    #     for ind in scale:
    #         x, y, z = ind
    #         x = x * self.config.side_length
    #         y = y * self.config.side_length
    #         z = z * self.config.depth
    #         # print(x, y, z)
    #         list_ind = z.long() * depth + y * self.config.side_length + x.long()
    #         # print(list_ind)
    #         # list_ind = y * 100 + x
    #         list_index.append(list_ind)
    #     list_index = torch.LongTensor(list_index)
    #     # print(max(list_index))
    #     return list_index
    
# data = torch.load('/home/zhouge/Downloads/neural_renderer/examples/data/logo/human1.pkl')
# logo_scale = data['logo_scale']
# logo_scale = data['logo_scale'].long()
# map = torch.zeros(101, 101)
# for scale in logo_scale:
#     map[scale[0], scale[1]] = 1
# print(map)
# map = map.numpy()
# from matplotlib import pyplot as plt
#
# plt.imshow(map)
# plt.show()
# map1 = torch.ones(100, 100).numpy()
# plt.imshow(map1)
# plt.show()
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
