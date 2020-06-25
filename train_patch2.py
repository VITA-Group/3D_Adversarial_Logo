"""
Training code for Adversarial patch training


"""

import subprocess

import neural_renderer as nr
from tensorboardX import SummaryWriter
from torch import autograd
from tqdm import tqdm
import pdb
import patch_config
from load_data import *
import argparse
# torch.cuda.set_device(5)


class RenderModel(nn.Module):
    def __init__(self):
        super(RenderModel, self).__init__()
        
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer
    
    def forward(self, camera_distance, elevation, vertices, faces, textures, angle):
        # vertices = mesh['logo_vertices'].cuda()
        # faces = mesh['logo_faces'].cuda()
        #
        # textures = self.textures.unsqueeze(0)
        self.renderer.eye = nr.get_points_from_angles(camera_distance, elevation, angle)
        logo_images, _, _ = self.renderer(vertices, faces,
                                          textures)  # [batch_size, RGB, image_size, image_size]
        
        image = torch.flip(logo_images, [-1])
        
        return image


class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.render_model = RenderModel()
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()  # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()
        
        self.writer = self.init_tensorboard(mode)
    
    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()
    
    def train(self,filename, camera_distance, elevation, angle, scale):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        
        batch_size = self.config.batch_size
        n_epochs = 10000
        max_lab = 14
        
        time_str = time.strftime("%Y%m%d-%H%M%S")
        
        # Generate stating point
        # adv_patch_cpu = self.generate_patch("gray")
        # adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")
        
        # adv_patch_cpu.requires_grad_(True)
        img_size = self.darknet_model.height
        train_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                         shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=10)
        # mesh_names = fnmatch.filter(os.listdir(os.path.join(data_dir)), '*.pkl')
        # mesh_paths = []
        # for mesh_name in mesh_names:
        #     mesh_paths.append(os.path.join(data_dir, mesh_name))
        # mesh_path = 'data/human2.pkl'
        mesh = torch.load(filename)
        vertices = mesh['logo_vertices'].cuda()
        faces = mesh['logo_faces'].cuda()
        textures = nn.Parameter(mesh['logo_textures'].unsqueeze(0).cuda())
        
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')
        
        optimizer = optim.Adam([textures], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        
        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = self.render_model(camera_distance, elevation, vertices, faces, textures, angle)
                    adv_patch = F.interpolate(adv_patch, (self.config.patch_size, self.config.patch_size),
                                              mode='bilinear').squeeze(0)
                    # adv_patch = adv_patch_cpu.cuda()
                    img_size = self.darknet_model.height
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, scale, do_rotate=True,
                                                         rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))
                    
                    img = p_img_batch[0, :, :, ]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    img.save('data/result_origin.jpg')
                    # img.show()
                    
                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)
                    
                    nps_loss = nps * 0.01
                    tv_loss = tv * 2.5
                    det_loss = torch.mean(max_prob)
                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    
                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch.data.clamp_(0, 1)  # keep patch in image range
                    
                    bt1 = time.time()
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        
                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                        
                        # self.writer.add_image('patch', adv_patch, iteration)
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_nps_loss = ep_nps_loss / len(train_loader)
            ep_tv_loss = ep_tv_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)
            
            # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # plt.imshow(im)
            # plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')
            
            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1 - et0)
                # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                # plt.imshow(im)
                # plt.show()
                # im.save("saved_patches/patchnew1.jpg")
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()
    
    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))
        
        return adv_patch_cpu
    
    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()
        
        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str,
                        default=os.path.join(data_dir, 'human2.pkl'))
    parser.add_argument('-m', '--mode', type=str, default='paper_obj')
    # parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'result3.jpg'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-angle', '--camera_angle', type=int, default=180)
    # parser.add_argument('-ind', '--index', type=int, default=0)
    parser.add_argument('-e', '--evaluation', type=float, default=0)
    parser.add_argument('-d', '--camera_distance', type=float, default=2.)
    parser.add_argument('-s', '--scale', type=float, default=1.5)
    args = parser.parse_args()
    filename = args.filename_input
    # other settings
    distance = args.camera_distance
    elevation = args.evaluation
    angle = args.camera_angle
    scale = args.scale
    trainer = PatchTrainer(args.mode)
    trainer.train(filename, distance, elevation, angle, scale)
    torch.cuda.set_device(args.gpu)

if __name__ == '__main__':
    main()
