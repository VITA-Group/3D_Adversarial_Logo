from darknet import Darknet
from utils import *


def clip(img):
    h, w = img.shape[:2]
    if h == w:
        return img
    else:
        if h > w:
            clip = int((h - w) / 2)
            return img[clip: w + clip, :, :]
        if h < w:
            clip = int(w - h)
            return img[:, clip:, :]


def detect_cv2(cfgfile, weightfile, imgdir, output_dir, target):
    import cv2
    # output_dir = 'background'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
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
    # pdb.set_trace()
    for folder in os.listdir(imgdir):
        folder_path = os.path.join(imgdir, folder)
        for imgfile in os.listdir(folder_path):
            count = 0
            if imgfile.endswith('.jpg') or imgfile.endswith('.png') or imgfile.endswith('.jpeg'):
                imgpath = os.path.join(folder_path, imgfile)
                # print(imgfile)
                img = cv2.imread(imgpath)
                # print('original',img.shape)
                
                # print(img.shape)
                if img is None:
                    continue
                img = clip(img)
                # print('clipped',img.shape)
                sized = cv2.resize(img, (m.width, m.height))
                # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
                
                start = time.time()
                boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                finish = time.time()
                print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
                for box in boxes:
                    if box[6] == int(target):
                        count += 1
                if not count:
                    output_path = os.path.join(output_dir, imgfile)
                    cv2.imwrite(output_path, sized)
                
                # class_names = load_class_names(namesfile)
                # plot_boxes_cv2(img, boxes, savename='predicitons.jpg', class_names=class_names)


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 6:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        output_dir = sys.argv[4]
        target = sys.argv[5]
        detect_cv2(cfgfile, weightfile, imgfile, output_dir, target)
        # detect_cv2(cfgfile, weightfile, imgfile)
        # detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        # detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
