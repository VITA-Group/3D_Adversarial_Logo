import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

img = Image.open('/home/zhouge/Downloads/example1.jpg').convert('RGB')
img = transforms.ToTensor()(img)
# img = torch.from_numpy(img.astype(np.float32) / 255.)
# h, w = img.shape[:2]

# img = img.permute(2, 0, 1)

# dim = (int(w / 2), int(w / 2), int(h / 2), int(h / 2))
# img = F.pad(img, dim, 'constant', value=127)
img_ref = Image.open('/home/zhouge/Pictures/street.jpg').convert('RGB')


# img_ref = transforms.ToTensor(img_ref)

#

# def pad(img, image_ref, img_size):
#     # street_img = street_img.transpose(1, 2, 0)
#     w, h = image_ref.size
#     # img_size = max(w, h)
#     i_h, i_w = img.shape[1:]
#
#     h_top = int((img_size - i_h) / 2) if (img_size - i_h) % 2 == 0 else int((img_size - i_h) / 2) + 1
#     h_bottom = int((img_size - i_h) / 2)
#     w_top = int((img_size - i_w) / 2) if (img_size - i_h) % 2 == 0 else int((img_size - i_h) / 2) + 1
#     w_bottom = int((img_size - i_w) / 2)
#     # TODO:padiing img
#     dim = (w_top, w_bottom, h_top, h_bottom)
#     img = F.pad(img, dim, 'constant', value=0.)
#     if w == h:
#         padded_img = image_ref
#     else:
#         dim_to_pad = 1 if w < h else 2
#         if dim_to_pad == 1:
#             padding = (h - w) / 2
#             padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
#             padded_img.paste(image_ref, (int(padding), 0))
#
#         else:
#             padding = (w - h) / 2
#             padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
#             padded_img.paste(image_ref, (0, int(padding)))
#     transform = transforms.Compose([transforms.Resize((img_size, img_size)),
#                                     transforms.ToTensor()])
#
#     padded_img = transform(padded_img)
#
#     padded_img = torch.where(img == 0., padded_img, img)
#
#     return padded_img

def pad(img, image_ref, img_size):
    # street_img = street_img.transpose(1, 2, 0)
    
    # img_size = max(w, h)
    i_h, i_w = img.shape[1:]
    
    h_top = int((img_size - i_h) / 2) if (img_size - i_h) % 2 == 0 else int((img_size - i_h) / 2) + 1
    h_bottom = int((img_size - i_h) / 2)
    w_top = int((img_size - i_w) / 2) if (img_size - i_h) % 2 == 0 else int((img_size - i_h) / 2) + 1
    w_bottom = int((img_size - i_w) / 2)
    # TODO:padiing img
    dim = (w_top, w_bottom, h_top, h_bottom)
    img = F.pad(img, dim, 'constant', value=0.)
    
    # pad reference image
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
    
    padded_img = transform(padded_img)
    
    padded_img = torch.where(img == 0., padded_img, img)
    
    return padded_img


image = pad(img, img_ref, 416)
# image = image.data.numpy()
# image = image.transpose((1, 2, 0))
# image = image.permute((1, 2, 0))
print(image.shape)
image = transforms.ToPILImage()(image)
image=image.convert('RGB')
print(image.size)
# cv2.imwrite('/home/zhouge/Pictures/pad.jpg', image * 255)
image.save('/home/zhouge/Pictures/pad.jpg')
img_ref.save('/home/zhouge/Pictures/img_ref.jpg')