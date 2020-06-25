"""
Training code for Adversarial patch training


"""
import subprocess

from tensorboardX import SummaryWriter
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm

import patch_config
from load_data import *
from render_model import RenderModel

torch.cuda.set_device(5)
# os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
torch.backends.cudnn.benchmark = True
from BackgroundDataset import BackgroundDataset


class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()
        for p in self.darknet_model.parameters():
            p.requires_grad = False
        self.render_model = RenderModel(self.config).cuda()
        # self.patch_applier = PatchApplier().cuda()
        # self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
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
    
    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        
        img_size = self.render_model.darknet_model.height
        batch_size = 16
        n_epochs = 10000
        # max_lab = 14
        # time_str = time.strftime("%Y%m%d-%H%M%S")
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        bk_gnd_dir = os.path.join(current_dir, 'street')
        data_dir = os.path.join(current_dir, 'data')
        save_path = os.path.join(current_dir, 'data/model/model_params.pkl')
        # mesh_dir = os.path.join(data_dir, 'human.obj')
        # filename_ref = os.path.join(data_dir, 'street.jpg')
        # filename_logo = os.path.join(data_dir, 'logo_index.pickle')
        # render_model = Model(filename_obj, filename_ref, filename_logo, img_size)
        if os.path.exists(save_path):
            self.render_model.load_state_dict(torch.load(save_path), strict=False)
        # render_model = nn.DataParallel(render_model)
        
        # background iamges
        background_data = DataLoader(BackgroundDataset(bk_gnd_dir, img_size,
                                                       shuffle=True),
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=10)
        
        # load mesh data
        mesh_names = fnmatch.filter(os.listdir(os.path.join(data_dir)), '*.pkl')
        mesh_paths = []
        for mesh_name in mesh_names:
            mesh_paths.append(os.path.join(data_dir, mesh_name))
        # mesh_data = DataLoader(MeshDataset(data_dir, shuffle=True),
        #                        batch_size=1,
        #                        shuffle=True,
        #                        num_workers=10)
        # render_model.cuda()
        self.epoch_length = len(background_data)
        print(f'One epoch is {self.epoch_length}')
        
        # print(render_model.state_dict())
        # print(render_model.parameters())
        # for name, param in render_model.named_parameters():
        #     if param.requires_grad:
        # print(name)
        # print(param.shape)
        optimizer = optim.Adam(self.render_model.parameters(), lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        
        et0 = time.time()
        
        # pdb.set_trace()
        # print(next(enumerate(mesh_data)))
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_loss = 0
            bt0 = time.time()
            angle = 180
            # num_images = 10
            # grad_param_saved = OrderedDict()
            
            for i_batch, bg_data in tqdm(enumerate(background_data), desc=f'Running epoch {epoch}'):
                bg_data = bg_data.cuda()
                # for m_batch, mesh_path in tqdm(enumerate(mesh_paths), desc=f'Running epoch {m_batch}'):
                for m_batch, mesh_path in tqdm(enumerate(mesh_paths)):
                    # print(mesh_path)
                    mesh = torch.load(mesh_path)
                    # TODO: bg_data, mesh_data process
                    with autograd.detect_anomaly():
                        # pdb.set_trace()
                        # img_batch = self.render_model.forward(mesh, bg_data, angle)
                        # print(img_batch)
                        output = self.render_model(mesh, bg_data, angle)
                        # print(img_batch.size())
                        # img_batch = img_batch.cuda()
                        
                        # p_img_batch = F.interpolate(img_batch, (self.darknet_model.height, self.darknet_model.width))
                        # img = p_img_batch.detach().cpu().data[0, :, :, ]
                        # img = transforms.ToPILImage()(img)
                        # img.save('data/result.jpg')
                        # output = self.darknet_model(p_img_batch)
                        # calc obj loss
                        max_prob = self.prob_extractor(output)
                        det_loss = torch.mean(max_prob)
                        loss = det_loss
                        ep_det_loss += det_loss.detach().cpu().numpy()
                        ep_loss += loss.item()
                        
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        
                        optimizer.step()
                        # print(render_model.grad_textures)
                        bt1 = time.time()
                        if i_batch % 2 == 0:
                            iteration = self.epoch_length * epoch + i_batch
                            
                            # self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                            # self.writer.add_scalar('misc/epoch', epoch, iteration)
                            # self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                            
                            # self.writer.add_image('patch', img, iteration)
                        if i_batch + 1 >= self.epoch_length:
                            print('\n')
                        else:
                            # del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                            # del output, max_prob, det_loss, p_img_batch, img_batch, loss
                            del output, max_prob, det_loss, loss, mesh
                            
                            torch.cuda.empty_cache()
                        bt0 = time.time()
                        bt0 = time.time()
                # grad_param_saved['grad_textures'] = render_model.state_dict()['grad_textures']
                if epoch % 10 == 0:
                #     torch.save(grad_param_saved, save_path)
                    torch.save(self.render_model.state_dict(), save_path)
            
            et1 = time.time()
            ep_loss = ep_loss / self.epoch_length
            ep_det_loss = ep_det_loss / self.epoch_length
            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                
                print('EPOCH TIME: ', et1 - et0)
                # del output, max_prob, det_loss, p_img_batch, loss
                del output, max_prob, det_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)
    
    trainer = PatchTrainer(sys.argv[1])
    trainer.train()


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
