import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from skimage import io
import skimage.transform as sktransform
import skimage.util as skutil
import matplotlib.pylab as plt
import json

from config import opt

class Transform():
    
    def __init__(self, min_size=opt.min_size, max_size=opt.max_size):
        self.min_size = min_size
        self.max_size = max_size
        
    def __call__(self, data):
        img, bbox = data
        
        img = img.transpose(2, 0, 1)
        img = img/255.
        
        # img normalize
        img = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(torch.from_numpy(img).float()).numpy().astype(np.float32)
        
        # img resize
        c, h, w = img.shape
        smin, smax = self.min_size/min(h, w), self.max_size/max(h, w)
        scale = min(smin, smax)
        img = sktransform.resize(img, (c, h*scale, w*scale))
        
        # bbox shift and resize
        bbox = bbox * scale
        bbox = bbox[:, (1, 0, 3, 2)]
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
        
        # random pad for mini-batch training
#         _, h, w = img.shape
#         diff = max(h, w) - min(h, w)
#         half1 = np.random.randint(0, diff+1)
#         half2 = diff - half1
        
#         if h > w:
#             img = skutil.pad(img, ((0,0),(0,0),(half1, half2)),
#                              mode='constant', constant_values=0.5)
#             bbox[:, 1] += half1
#             bbox[:, 3] += half1
#         elif h < w:
#             img = skutil.pad(img, ((0,0),(half1, half2),(0,0)),
#                              mode='constant', constant_values=0.5)
#             bbox[:, 0] += half1
#             bbox[:, 2] += half1
            
#         img = sktransform.resize(img, (c, opt.batch_spatial_size,
#                                        opt.batch_spatial_size))
#         p_scale = opt.batch_spatial_size / max(h, w)
#         bbox = bbox * p_scale
#         scale = scale * p_scale
        return img, bbox, scale


class TrainValSet(Dataset):

    def __init__(self, root, info='digitStruct.json'):
        self.root = root
        info_path = root+'/'+info
        with open(info_path, 'r') as f:
            self.data_info = json.load(f)
        self.transform = Transform()
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = self.root+'/'+self.data_info[idx]['name']
        img = io.imread(img_path)
        
        bbox = np.asarray(self.data_info[idx]['bbox']).astype(np.float32)
        label = np.asarray(self.data_info[idx]['label']).astype(np.int32)
        label[label==10]=0 # 0 is labeled as 10 in svhn

        img, bbox, scale = self.transform((img, bbox))

        return img, bbox, label, scale


def denormalize(img):
    return (img * 0.225 + 0.45).clip(0, 1) 


def show_item(item):
    img, bbox, label, scale = item
    img = img.transpose(1, 2, 0)
    img = denormalize(img)
    
    title = 'Label: '+ str(label)+" shape: " + str(img.shape)
    
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    ax = plt.gca()
    for i in range(bbox.shape[0]):
        xy = (bbox[i, 1], bbox[i, 0])
        w, h = bbox[i, 3]-bbox[i, 1], bbox[i, 2]-bbox[i, 0]
        rect = plt.Rectangle(xy, w, h, fill=False, edgecolor = 'red',linewidth=1)
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()