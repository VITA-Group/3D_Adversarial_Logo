import sys
import time
from PIL import Image, ImageDraw
from utils import *
from darknet import Darknet
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
import pdb
def detect(cfgfile, weightfile, imgdir):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()
    folder_name = ['clean', 'logo']
#    pdb.set_trace()
    for folder in folder_name:
        imgpath = imgdir + '/' + folder
        for imgfile in os.listdir(imgpath):
            if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
#            print(folder)
#            print(imgfile)
            
                name = os.path.splitext(imgfile)[0]
#            print(name)
                img = Image.open(imgpath + '/' + imgfile).convert('RGB')
                sized = img.resize((m.width, m.height))
    
    # Simen: niet nodig om dit in loop te doen? 
    #for i in range(2):
                start = time.time()
                # pdb.set_trace()
                boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                # print(boxes)
                finish = time.time()
    
    #if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))
                savename = 'predictions/{}_{}.jpg'.format(name, folder)
                class_names = load_class_names(namesfile)
                plot_boxes(img, boxes, savename, class_names)


def detect_mtl(cfgfile, weightfile, imgdir):
    m = Darknet(cfgfile)
    
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    
    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()
    folder_name = ['clean', 'logo']
    #    pdb.set_trace()
    image_file = torch.load(imgdir)
    print(image_file.shape)
    transform = transforms.ToPILImage()
    for ii, img in enumerate(image_file):
        start = time.time()
        boxes = do_detect(m, img, 0.5, 0.4, use_cuda)
        finish = time.time()
        print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
        savename = 'predictions/{}.png'.format(ii)
        class_names = load_class_names(namesfile)
        img = transform(img)
        plot_boxes(img, boxes, savename, class_names)
        
        
        

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()
    # print(torch.load(imgfile)[1].shape)
    # image_file = torch.load(imgfile)[1].transpose(1, 2, 0)*255
    
    # print(image_file.shape)
    # img = cv2.imread(imgfile)
    # sized = cv2.resize(img, (m.width, m.height))
    # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    # for folder in os.listdir(imgfile):
    #     folder_path = os.path.join(imgfile, folder)
    bk_image = Image.open('1.jpg')
    transform = transforms.ToTensor()
    bk_image = transform(bk_image)
    for file_name in os.listdir(imgfile):
        file_path = os.path.join(imgfile, file_name)
        file = torch.load(file_path)
        mtls = file['mtl']
        for mi, mtl in enumerate(mtls):
            print(mtl.shape)
            mtl_image = torch.FloatTensor(mtl).squeeze(0)
            print(bk_image.dtype)
            print(mtl_image.dtype)
            mtl_image = F.pad(mtl_image, (80, 80, 160, 0), 'constant', value=0.)
            det_image = torch.where(mtl_image == 0, bk_image, mtl_image)
            for i in range(2):
                boxes = do_detect(m, det_image, 0.8, 0.4, use_cuda)
                print(boxes)

            det_image = det_image.data.numpy().transpose(1, 2, 0)
            det_image = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
            savename = 'predictions/image/' + file_name + str(mi) + '.png'
            class_names = load_class_names(namesfile)
            plot_boxes_cv2(det_image * 255, boxes, savename=savename, class_names=class_names)
            # mtl_name = 'mtls/' + file_name + str(mi) + '.png'
            # mtl = mtl * 255
            # mtl = cv2.cvtColor(mtl, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(mtl_name, mtl)
        # image = file['image']
        # for ii, img in enumerate(image):
        # # m.eval()
        # # print(file.shape)
        # # file = torch.from_numpy(file.transpose(0, 3, 1, 2))
        # # output = m(file.cuda())
        # # dis_loss(output, m.num_classes, m.anchors, m.num_anchors, target_id=0, only_objectness=1,
        # #          validation=False)
        # # for ii, img in enumerate(file):
        #     # print(img.shape)
        #     # img = img
        #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     for i in range(2):
        #         start = time.time()
        #         boxes = do_detect(m, img, 0.65, 0.4, use_cuda)
        #         print(boxes)
        #         finish = time.time()
        #         if i == 1:
        #             print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
        #     img = img.data.numpy().transpose(1,2,0)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     savename = 'predictions/image/' + file_name + str(ii) + '.png'
        #     class_names = load_class_names(namesfile)
        #     plot_boxes_cv2(img*255, boxes, savename=savename, class_names=class_names)
    # for ii, img in enumerate(image_file):
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     for i in range(2):
    #         start = time.time()
    #         boxes = do_detect(m, img, 0.5, 0.4, use_cuda)
    #         print(boxes)
    #         finish = time.time()
    #         if i == 1:
    #             print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
    #     # img = img.numpy().transpose(1,2,0)
    #     class_names = load_class_names(namesfile)
    #     plot_boxes_cv2(img, boxes, savename='predictions/{}.png'.format(ii), class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    
    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()
    
    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)




if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        # index = int(sys.argv[4])
        # detect(cfgfile, weightfile, imgfile)
        # detect_mtl(cfgfile, weightfile, imgfile)
        detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
