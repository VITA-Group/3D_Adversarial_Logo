import math
import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import pdb

def pad(img, pos):
    '''

    :param img:
    :param bk_image:
    :return: pasted img

    paste 2d img rasterized from mesh onto background imgs
    '''
    i_h, i_w = img.shape[2:]

    h_pad_len = 416 - i_h

    w_pad_len = 416 - i_w
    paste_imgs = []
    # paste
    # h_pos = rd.randint(0, h_pad_len)
    h_pos = int(h_pad_len * 0.75)

    w_pos = int(w_pad_len * 0.5)
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

plt.yticks([])
plt.xticks([])
plt.axis('off')

# texture_aug = torch.zeros()
def augment(img, number = 4):
    # pdb.set_trace()
    img = img.unsqueeze(0)
    i_h, i_w = img.shape[2:]
    size = torch.linspace(0.4, 1, 5)[0:]
    aug_imgs = []
    for scale in size:
        rots = torch.linspace(-20, 20, number)
        # print(rots)
        
        poses = torch.linspace(416, 0, number + 2)
        print(poses)
        board = torch.zeros(1, 3, 416, 416)
        for pi, (rot, pos) in enumerate(zip(rots, poses[2:])):
            print(poses[2:])
            # print(rot, pos)
            image = img.clone()
            pallete = torch.zeros(image.shape)
            channel = pi % 3
            pallete[:, channel, int(i_h * 0.6):, :] = 1.
            color_aug = torch.where(image != 0. , pallete, image)
            image = torch.where(color_aug != 0., color_aug, image)
            image = F.interpolate(image, (int(scale * i_h), int(scale * i_w)), mode='bilinear')
            angle = float(rot) * math.pi / 180
            theta = torch.tensor([
                [math.cos(angle), math.sin(-angle), 0],
                [math.sin(angle), math.cos(angle), 0]
            ], dtype=torch.float)
            # blank = torch.ones(target.shape)
            grid = F.affine_grid(theta.unsqueeze(0), image.size())
            output = F.grid_sample(image, grid)
            output = pad(output, pos)
            board = torch.where(output == 0., board, output)
        plt.imshow(board[0].numpy().transpose(1, 2, 0))
        plt.savefig('/home/zhouge/Documents/aug{}.pdf'.format(scale), bbox_inches='tight')
        aug_imgs.append(board)

# def logo_augment(logo):
#     logo

cur_dir = os.path.dirname(os.path.realpath(__file__))
image_file = os.path.join(cur_dir, 'image/mtl')
# for file_name in os.listdir(image_file):
#     file_path = os.path.join(image_file, file_name)
#
#     target, logo, syn = torch.load(file_path)
#     np_blank = np.ones(target.shape, dtype=np.float32)
#     # target = np.where(target == 0., np_blank, target)
#     # plt.imshow(target[0].transpose(1,2,0))
#     # plt.savefig('/home/zhouge/Documents/target.pdf', bbox_inches = 'tight')
#     # logo = np.where(logo == 0., np_blank, logo)
#     # plt.imshow(logo[0].transpose(1,2,0))
#     # plt.savefig('/home/zhouge/Documents/logo.pdf', bbox_inches = 'tight')
#     # transform = transforms.ToTensor()
#     # image = transform(image)
#     blank = torch.ones(target.shape)
#
#     target = torch.from_numpy(target).squeeze(0)
#
#     logo = torch.from_numpy(logo).squeeze(0)
#     syn = torch.from_numpy(syn).squeeze(0)
#     noise = torch.FloatTensor(logo.size()).uniform_(-1, 1)
#     target = torch.where(target == 0., blank, target).squeeze(0)
#     syn = torch.where(syn == 0., blank, syn).squeeze(0)
#     noise_image = torch.add(noise, logo)
#     noise_image = torch.where(logo == 0., logo, noise_image).squeeze(0)
#     logo = torch.where(logo == 0., blank, logo).squeeze(0)
#     noise_image = torch.where(noise_image == 0., blank, noise_image).squeeze(0)
#     # plt.imshow(target.numpy().transpose(1,2,0))
#     # plt.savefig('/home/zhouge/Documents/target.pdf', bbox_inches = 'tight')
#     # plt.imshow(logo.numpy().transpose(1,2,0))
#     # plt.savefig('/home/zhouge/Documents/logo.pdf', bbox_inches = 'tight')
#     # image = torch.where(logo == 0., target, logo)
#
#     # augment(image)
#
#     # aug_img = copy(image)
#     # aug_img[2, 150:, :] = 1.
#     # plt.imshow(aug_img.numpy().transpose(1, 2, 0))
#     # plt.savefig('/home/zhouge/Documents/aug_img.pdf', bbox_inches = 'tight')
#     # image = torch.where(image == 0., blank, image.unsqueeze(0)).squeeze(0)
#     # plt.imshow(image.numpy().transpose(1,2,0))
#     # plt.savefig('/home/zhouge/Documents/merge.pdf', bbox_inches = 'tight')
#     angle = -30 * math.pi / 180
#     theta = torch.tensor([
#         [math.cos(angle), math.sin(-angle), 0],
#         [math.sin(angle), math.cos(angle), 0]
#     ], dtype=torch.float)
#     # blank = torch.ones(target.shape)
#     grid = F.affine_grid(theta.unsqueeze(0), syn.unsqueeze(0).size())
#     output = F.grid_sample(syn.unsqueeze(0), grid)
#
#     output = torch.where(output == 0., blank, output)
#     new_img_torch = output[0]
#     # print(new_img_torch)
#     plt.imshow(new_img_torch.numpy().transpose(1, 2, 0))
#     plt.savefig('/home/zhouge/Documents/{}_{}.pdf'.format(file_name, 'aug'), bbox_inches='tight')
#     plt.imshow(syn.numpy().transpose(1, 2, 0))
#     plt.savefig('/home/zhouge/Documents/{}_{}.png'.format(file_name, 'syn'), bbox_inches='tight')
#     plt.imshow(target.numpy().transpose(1, 2, 0))
#     plt.savefig('/home/zhouge/Documents/{}_{}.png'.format(file_name, 'target'), bbox_inches='tight')
#     plt.imshow(logo.numpy().transpose(1, 2, 0))
#     plt.savefig('/home/zhouge/Documents/{}_{}.png'.format(file_name, 'logo'), bbox_inches='tight')
#     plt.imshow(noise_image.numpy().transpose(1, 2, 0))
#     plt.savefig('/home/zhouge/Documents/{}_{}.png'.format(file_name, 'noise_logo'), bbox_inches='tight')
#         # plt.savefig('/home/zhouge/Documents/aug_img.jpg', bbox_inches='tight')
#         # plt.show()

img_dir = '/home/zhouge/Downloads/park_adv/selection'
def image_process(imgdir):

    import cv2
    count = 0
    # for i in range(1, 3 + 1):
    #     for j in range(2, 6 + 1):
    #         count += 1
    #         file_name = '{}_{}.png'.format(j, i)
    #         print(file_name)
    #         print(i * j - i + j)
    #         file_path = os.path.join(imgdir, file_name)
    #         img = cv2.imread(file_path)
    #         # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #         # plt.savefig(img_dir + '/{}_{}'.format(j, i) + '.pdf', bbox_inches='tight')
    #         plt.subplot(3, 5, count)
    #         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #         plt.axis('off')
    # plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    # plt.savefig(img_dir + '/mtl.pdf', bbox_inches='tight')
    for file in os.listdir(imgdir):
        filename = file.split('.')[0]
        file_path = os.path.join(imgdir, file)
        # print(file_path)
        # file_name = '{}_{}.png'.format(j, i)
        # print(file_name)
        # print(i * j - i + j)
        # file_path = os.path.join(imgdir, file_name)
        img = cv2.imread(file_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.savefig(imgdir + '/' + filename + '.pdf', bbox_inches='tight')

# image_process(img_dir)


mtl_dir = '/home/zhouge/Documents/3dmesh'


def prepare_mtl(datadir):
    import numpy as np
    import cv2
    mtldir = datadir + '/mtl'
    count = 0
    # for i in range(1, 3 + 1):
    #     for j in range(2, 6 + 1):
    #         count += 1
    #         file_name = '{}_{}.png'.format(j, i)
    #         print(file_name)
    #         print(i * j - i + j)
    #         file_path = os.path.join(imgdir, file_name)
    #         img = cv2.imread(file_path)
    #         # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #         # plt.savefig(img_dir + '/{}_{}'.format(j, i) + '.pdf', bbox_inches='tight')
    #         plt.subplot(3, 5, count)
    #         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #         plt.axis('off')
    # plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    # plt.savefig(img_dir + '/mtl.pdf', bbox_inches='tight')
    blank = np.ones((256, 256, 3), dtype=np.uint8) * 255
    for file in os.listdir(mtldir):
        filename = file.split('.')[0]
        file_path = os.path.join(mtldir, file)
        print(file_path)
        # file_name = '{}_{}.png'.format(j, i)
        # print(file_name)
        # print(i * j - i + j)
        # file_path = os.path.join(imgdir, file_name)
        img = cv2.imread(file_path)
        print(img.dtype)
        img = np.where(img == 0, blank, img)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.savefig(datadir + '/pdf/' + filename + '.png', bbox_inches='tight')
 
        
prepare_mtl(mtl_dir)