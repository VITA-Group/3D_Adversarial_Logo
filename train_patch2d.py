"""
Training code for Adversarial patch training


"""
import subprocess
import copy
from tensorboardX import SummaryWriter
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import cv2 as cv
# import patch_config
from load_data import *
import torch.optim as optim
from render_model2d import *
# torch.cuda.set_device(4)
from collections import OrderedDict
import pdb
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
torch.backends.cudnn.benchmark = True
from BackgroundDataset import BackgroundDataset


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

class NmaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls):
        super(NmaxProbExtractor, self).__init__()
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
    
    
class PatchTrainer(nn.Module):
    def __init__(self, config):
        super(PatchTrainer, self).__init__()
        self.config = config
        # if self.config.consistent:
        #     self.grad_textures = nn.Parameter(torch.full((self.config.depth * self.config.width * self.config.height,
        #                                                   3), 0.5))
        # else:
        #     self.grad_textures = nn.Parameter(torch.full((self.config.depth * self.config.width * self.config.height,
        #                                                   4, 4, 4, 3), 0.5))
        
        # self.darknet_model = Darknet(self.config.cfgfile)
        # self.darknet_model.load_weights(self.config.weightfile)
        # self.darknet_model = self.darknet_model.eval().cuda()
        # for p in self.darknet_model.parameters():
        #     p.requires_grad = False
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_darknet_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()
        for p in self.darknet_model.parameters():
            p.requires_grad = False
        print('img-szie', self.darknet_model.img_size)
        # self.Renderer = RenderModel(self.config)
        # self.Adv_train = AdvTrain(self.config, self.darknet_model.width, self.darknet_model.height)
        self.Adv_patch = AdvPatch(self.config, self.darknet_model.img_size, self.darknet_model.img_size)
        # self.Adv_test = AdvTest(self.config, self.darknet_model.width, self.darknet_model.height)
        self.prob_extractor = NmaxProbExtractor(0, 80).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.image_size).cuda()
        self.total_variation = TotalVariation().cuda()
        self.medianpooler = MedianPool2d(3, same=True)
        # if self.config.cuda is not -1:
        #     torch.cuda.set_device(self.config.cuda)
        #     self.device = torch.device('cuda')
        # else:
        #     self.device = torch.device('cpu')
        # self.patch_applier = PatchApplier().cuda()
        # self.patch_transformer = PatchTransformer().cuda()
        # self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        # self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        # self.total_variation = TotalVariation().cuda()
        
        # self.writer = self.init_tensorboard(mode)
    
    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self, universal_logo_cpu, mesh_paths, train_bg_data,
              epoch, train_mesh, logonum, angle_range, optimizer, scheduler, length):
        # load mesh data
        et0 = time.time()
        ep_dis_loss = 0
        ep_nps_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        if self.config.width != self.config.height:
            logo_info = 'H'
        else:
            logo_info = 'G'
        random.shuffle(mesh_paths)
        print(mesh_paths)
        count = [0 for i in range(len(mesh_paths))]
        for i_batch, bg_data in tqdm(enumerate(train_bg_data), total=self.epoch_length):
        
            for m_batch, mesh_path in tqdm(enumerate(mesh_paths), desc=f'Running epoch {epoch}', total=len(mesh_paths)):
                # count = 0
                m_name = mesh_path.split('/')[-1]
                bg_data = bg_data.cuda()
                mesh = torch.load(mesh_path)
            
                mesh_angle = mesh['mesh_angle']
                target_images = mesh['target_images']
                vertices, faces, logo_scale = mesh['{}k'.format(logonum)]
                # pdb.set_trace()
                for ai, angle in enumerate(range(mesh_angle - angle_range, mesh_angle + angle_range + 1)):
                    # print(ai, angle)
                    # TODO: bg_data, mesh_data process
                    # train_mesh = copy.copy(mesh)
                    with autograd.detect_anomaly():
                        universal_logo = universal_logo_cpu.cuda()
                        if self.config.train_patch:
                            training_images = self.Adv_patch(universal_logo, logo_info, target_images[20 - ai], bg_data)
                            tv = self.total_variation(universal_logo)
                        else:
                            training_images, logo_image = self.Adv_train(universal_logo, vertices, faces,
                                                             logo_scale,
                                                             target_images[20 - ai],
                                                             bg_data, angle)
                            tv = self.total_variation(logo_image[0])
                        # result check
                        img = training_images.detach().cpu().data[0, :, :, ]
                        img = transforms.ToPILImage()(img)
                        img.save('data/result{}.jpg'.format(m_batch))
                        
                        output = self.darknet_model(training_images)
                        print('output-shape', output.shape)
                        max_prob = self.prob_extractor(output)
                        dis_loss = torch.mean(max_prob)
                        neg_count = calc_acc(output, self.darknet_model.num_classes,
                                             self.darknet_model.num_anchors, self.config.target)
                        
                        tv_loss = tv * 2.5
                        count[m_batch] += neg_count
                        loss = dis_loss + torch.sum(torch.max(tv_loss, torch.tensor(0.1).cuda()))
                        ep_dis_loss += dis_loss.detach().cpu().numpy()
                    
                        # ep_nps_loss += nps_loss.detach().cpu().numpy()
                        ep_tv_loss += torch.sum(tv_loss).detach().cpu().numpy()
                        ep_loss += loss.item()
                    
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        universal_logo_cpu.data.clamp_(0, 1)
                        # self.grad_textures.clamp_(min=1e-7, max=0.999999)
        
            if i_batch + 1 >= len(train_mesh):
                print('\n')
        
            else:
                del dis_loss, tv_loss, loss
            
                torch.cuda.empty_cache()
    
        for m_b, mesh_path in enumerate(mesh_paths):
            acc = count[m_b] / (length * (2 * angle_range + 1))
            mesh_n = mesh_path.split('/')[-1]
            print('\n')
            print(mesh_n.split('.')[0] + '-deceive acc:', acc.data)
        et1 = time.time()
        ep_loss = ep_loss / self.epoch_length / (2 * angle_range + 1) / len(mesh_paths)
        ep_dis_loss = ep_dis_loss / self.epoch_length / (2 * angle_range + 1) / len(mesh_paths)
        # ep_nps_loss = ep_nps_loss / self.epoch_length / (2 * angle_range + 1) / len(mesh_paths)
        ep_tv_loss = ep_tv_loss / self.epoch_length / (2 * angle_range + 1) / len(mesh_paths)
        scheduler.step(ep_loss)
        if True:
            print('  EPOCH NR: ', epoch),
            print('EPOCH LOSS: ', ep_loss)
            print('  DIS LOSS: ', ep_dis_loss)
            # print('  NPS LOSS: ', ep_nps_loss)
            print('  TV LOSS: ', ep_tv_loss)
            print('EPOCH TIME: ', et1 - et0)
            # del output, max_prob, det_loss, p_img_batch, loss
            del dis_loss, loss
            torch.cuda.empty_cache()
        
    # def test(self, universal_logo_cpu, mesh_paths, test_bg_data,
    #          epoch, test_mesh, logonum, angle_range, length):
    #     # random.shuffle(mesh_paths)
    #     print(mesh_paths)
    #     total_accs=[]
    #     for m_batch, mesh_path in tqdm(enumerate(mesh_paths), desc=f'Testing epoch {epoch}', total=len(test_mesh)):
    #         universal_logo = universal_logo_cpu.cuda()
    #         
    #         mesh = torch.load(mesh_path)
    # 
    #         mesh_angle = mesh['mesh_angle']
    #         target_images = mesh['target_images']
    #         vertices, faces, logo_index = mesh['{}k'.format(logonum)]
    #         universal_logo = self.medianpooler(universal_logo.unsqueeze(0)).squeeze(0)
    #         universal_logo = universal_logo.permute(1, 2, 0).contiguous().view(-1, 3)
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
    #         vertices = vertices.cuda()
    #         faces = faces.cuda()
    #         accs = []
    #         for ai, angle in enumerate(range(mesh_angle - angle_range, mesh_angle + angle_range + 1)):
    #             count = 0
    #             logo_image = self.Renderer.render(vertices, faces,
    #                                               adversarial_logo, angle)
    #             target_image = target_images[20 - ai].cuda()
    #             for i_batch, bg_data in tqdm(enumerate(test_bg_data), total=len(test_bg_data)):
    #                 bg_data = bg_data.cuda()
    #                 merge = torch.where(logo_image == 0., target_image, logo_image)
    #                 merge = merge.expand(bg_data.shape[0], -1, -1, -1)
    #                 pos = int((self.darknet_model.width - merge.shape[-1]) / 2)
    #                 testing_images = paste(merge, bg_data, 0, pos)
    # 
    #                 testing_images = F.interpolate(testing_images, (self.darknet_model.height,
    #                                                                 self.darknet_model.width),
    #                                                mode='bilinear')
    #                 output = self.darknet_model(testing_images)
    #                 neg_count = calc_acc(output, self.darknet_model.num_classes,
    #                                      self.darknet_model.num_anchors, self.config.target)
    #                 count += neg_count
    #             acc = count / length
    #             accs.append(acc)
    #         total_accs.append(accs)
    #     total_accs = torch.tensor(total_accs).float()
    #     print('angle-acc', torch.mean(total_accs, 0).data)
    #     print('\n')
    #     print('-test acc:', (torch.sum(total_accs)/torch.numel(total_accs)).data)

    # def test(self, universal_logo_cpu, mesh_paths, test_bg_data,
    #          epoch, test_mesh, logonum, angle_range, length):
    #     # random.shuffle(mesh_paths)
    #     print(mesh_paths)
    #     total_accs = []
    #     for m_batch, mesh_path in tqdm(enumerate(mesh_paths), desc=f'Testing epoch {epoch}', total=len(test_mesh)):
    #         universal_logo = universal_logo_cpu.cuda()
    #
    #         mesh = torch.load(mesh_path)
    #
    #         mesh_angle = mesh['mesh_angle']
    #         target_images = mesh['target_images']
    #         vertices, faces, logo_index = mesh['{}k'.format(logonum)]
    #         universal_logo = self.medianpooler(universal_logo.unsqueeze(0)).squeeze(0)
    #         universal_logo = universal_logo.permute(1, 2, 0).contiguous().view(-1, 3)
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
    #         vertices = vertices.cuda()
    #         faces = faces.cuda()
    #         accs = []
    #         for ai, angle in enumerate(range(mesh_angle - angle_range, mesh_angle + angle_range + 1)):
    #             count = 0
    #             logo_image = self.Renderer.render(vertices, faces,
    #                                               adversarial_logo, angle)
    #             target_image = target_images[20 - ai].cuda()
    #             for i_batch, bg_data in tqdm(enumerate(test_bg_data), total=len(test_bg_data)):
    #                 bg_data = bg_data.cuda()
    #                 merge = torch.where(logo_image == 0., target_image, logo_image)
    #                 merge = merge.expand(bg_data.shape[0], -1, -1, -1)
    #                 pos = int((self.darknet_model.width - merge.shape[-1]) / 2)
    #                 testing_images = paste(merge, bg_data, 0, pos)
    #
    #                 testing_images = F.interpolate(testing_images, (self.darknet_model.height,
    #                                                                 self.darknet_model.width),
    #                                                mode='bilinear')
    #                 output = self.darknet_model(testing_images)
    #                 neg_count = calc_acc(output, self.darknet_model.num_classes,
    #                                      self.darknet_model.num_anchors, self.config.target)
    #                 count += neg_count
    #             acc = count / length
    #             accs.append(acc)
    #         total_accs.append(accs)
    #     total_accs = torch.tensor(total_accs).float()
    #     print('angle-acc', torch.mean(total_accs, 0).data)
    #     print('\n')
    #     print('-test acc:', (torch.sum(total_accs) / torch.numel(total_accs)).data)
    
    def attack(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        # torch.cuda.set_device(self.config.cuda)
        img_size = self.darknet_model.img_size
        batch_size = self.config.batch_size
        n_epochs = self.config.train_epoch
        # max_lab = 14
        # time_str = time.strftime("%Y%m%d-%H%M%S")
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        train_bk_grd_dir = os.path.join(current_dir, self.config.train_data_path)
        test_bk_grd_dir = os.path.join(current_dir, self.config.test_data_path)
        data_dir = os.path.join(current_dir, self.config.data_path)
        save_path = os.path.join(current_dir, self.config.model_path)
        # mesh_dir = os.path.join(data_dir, 'human.obj')
        # filename_ref = os.path.join(data_dir, 'street.jpg')
        # filename_logo = os.path.join(data_dir, 'logo_index.pickle')
        # render_model = Model(filename_obj, filename_ref, filename_logo, img_size)
        # if os.path.exists(save_path) and self.config.restore_model:
        #     self.render_model.load_state_dict(torch.load(save_path), strict=False)
        # render_model = nn.DataParallel(render_model)
        
        # background images
        if self.config.consistent:
            universal_logo_cpu = torch.full((3, self.config.height, self.config.width), 0.5)
        else:
            universal_logo_cpu = torch.full((self.config.depth * self.config.width * self.config.height,
                                             4, 4, 4, 3), 0.5)
        universal_logo_cpu.requires_grad_(True)
        train_bg_data = DataLoader(BackgroundDataset(train_bk_grd_dir, img_size,
                                                     shuffle=True),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=10)
        test_bg_data = DataLoader(BackgroundDataset(test_bk_grd_dir, img_size,
                                                    shuffle=True),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=10)
        # load mesh data
        meshobj = 'human{}.pkl'
        train_mesh = self.config.train_mesh
        test_mesh = self.config.test_mesh
        logonum = self.config.logonum
        
        # train mesh initialise
        train_mesh_names = []
        if self.config.conventional:
            train_mesh_names = [meshobj.format(self.config.mesh_id)]
        else:
            for ind in train_mesh:
                train_mesh_names.append(meshobj.format(ind))
        # mesh_names = fnmatch.filter(os.listdir(os.path.join(data_dir)), '*.pkl')
        train_mesh_paths = []
        for mesh_name in train_mesh_names:
            train_mesh_paths.append(os.path.join(data_dir, mesh_name))
        # test mesh initialise
        test_mesh_names = []
        if self.config.conventional:
            test_mesh_names = [meshobj.format(self.config.mesh_id)]
        else:
            for ind in test_mesh:
                test_mesh_names.append(meshobj.format(ind))
        # mesh_names = fnmatch.filter(os.listdir(os.path.join(data_dir)), '*.pkl')
        test_mesh_paths = []
        for mesh_name in test_mesh_names:
            test_mesh_paths.append(os.path.join(data_dir, mesh_name))
        # mesh_data = DataLoader(MeshDataset(data_dir, shuffle=True),
        #                        batch_size=1,
        #                        shuffle=True,
        #                        num_workers=10)
        # render_model.cuda()
        self.epoch_length = len(train_bg_data)
        print(f'One epoch is {self.epoch_length}')
        
        # print(render_model.state_dict())
        # print(render_model.parameters())
        # for name, param in self.render_model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # print(param.shape)
        optimizer = optim.Adam([universal_logo_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
        
        
        angle_range = self.config.angle_range
        
        # pdb.set_trace()
        # print(next(enumerate(mesh_data)))
        for epoch in range(n_epochs):
            # universal_logo = universal_logo_cpu.cuda()
            self.train(universal_logo_cpu, train_mesh_paths, train_bg_data, epoch, train_mesh, logonum, angle_range,
                       optimizer, scheduler, 312)
            # universal_logo_cpu.data.clamp_(0, 1)
            universal_logo_image = universal_logo_cpu.data.contiguous().permute(1,2,0).numpy()
            cv.imwrite('data/pics/universal_logo.png', universal_logo_image * 255)
            # if epoch%3 == 0:
                # self.test(universal_logo_cpu, train_mesh_paths, test_bg_data,
                #           epoch, test_mesh, logonum, angle_range, 38)
                # self.test(universal_logo_cpu, test_mesh_paths, train_bg_data,
                #           epoch, test_mesh, logonum, angle_range, 312)
                # self.test(universal_logo_cpu, test_mesh_paths, test_bg_data,
                #           epoch, test_mesh, logonum, angle_range, 38)
                # if self.config.save_model:
                #     torch.save(self.render_model.state_dict(), save_path)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Environment Configuration
    # parser.add_argument('--cuda', type=int, default=0, help='If -1, use cpu; if >=0 use single GPU; if 2,3,4 for multi GPUS(2,3,4)')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--train_data_path', type=str, default='train')
    parser.add_argument('--test_data_path', type=str, default='test')
    parser.add_argument('--output_path', type=str, default='out/facades/')
    parser.add_argument('--model_path', type=str, default='data/model/model_params.pkl')
    
    # Model Configuration

    parser.add_argument('--train_mesh', type=list, default=['1', '2'])
    parser.add_argument('--test_mesh', type=list, default=['3'])
    # Train Configuration
    parser.add_argument('--resume_epoch', type=int, default=-1,
                        help='if -1, train from scratch; if >=0, resume and start to train')
    
    # Test Configuration
    parser.add_argument('--angle_range', type=int, default=10)
    parser.add_argument('--logonum', type=int, default=8)
    parser.add_argument('--mesh_id', type=int, default=1)
    parser.add_argument('--train_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_image', type=str, default='',
                        help='if is an image, only translate it; if a folder, translate all images in it')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--height', type=int, default=150)
    parser.add_argument('--depth', type=int, default=1)
    # experiment configs
    parser.add_argument('--conventional', action='store_true', help='activate conventional approach')
    parser.add_argument('--paper_mtl', action='store_true', help='prepare paper images')
    parser.add_argument('--save_model', action='store_true', help='save model parameters')
    parser.add_argument('--restore_model', action='store_true', help='restore model parameters')
    parser.add_argument('--consistent', action='store_true', help='logo textures consistent')
    parser.add_argument('--train_patch', action='store_true', help='train 2D patch')
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--d', '--camera_distance', type=float, default=2.)
    parser.add_argument('--e', '--evaluation', type=float, default=0.)
    parser.add_argument('--cfgfile', type=str, default='config/yolov3.cfg')
    parser.add_argument('--weightfile', type=str, default="weights/yolov3.weights")
    parser.add_argument('--printfile',  type=str, default="non_printability/30values.txt")

    parser.add_argument('--start_learning_rate', type=float, default=0.03)

    # main function
    config = parser.parse_args()
    # torch.cuda.set_device(config.cuda)
    trainer = PatchTrainer(config)
 
    trainer.attack()


if __name__ == '__main__':
    main()

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
