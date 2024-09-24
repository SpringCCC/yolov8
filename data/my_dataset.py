from torch.utils.data.dataset import Dataset
from springc_utils import *
from collections import deque
from data.convert import *

class YOLODataset(Dataset):

    def __init__(self, img_info, config_yaml, class_name, imgsz, box_format='xywh') -> None:
        self.imgsz = imgsz
        self.imgs_path = img_info['img_info']
        self.boxs = img_info['boxs'] # (N, 5) 5: x1, y1, x2, y2, cls ,如果不符合格式，在输入YOLODataset之前处理
        self.data_queue = []
        self.config = type('TrainingConfig', (object,), config_yaml)
        self.n_class = len(class_name)
        self.class_name = class_name
        self.epoch_now = -1
        self.stop_mosaic_epoch = self.config.stop_mosaic_epoch
        self.n_imgs = len(self.imgs_path)
        self.box_format = box_format
        self.img_buffer, self.box_buffer = [None] * self.n_imgs, [None] * self.n_imgs
        self.buffer_idx = deque()
        self.max_buffer_num = self.config.max_buffer_num
        self.hsv = RandomHSV()
        self.mosaic = Mosaic()
        



    def __getitem__(self, i):
        img_path, box = self.load_image(i)
        if self.epoch_now < self.stop_mosaic_epoch and self.config.mosaic_prob>0:
            selected_ids = random.choice(self.buffer_idx, k=3)
            selected_ids.append(i)
            infos = []
            for j in selected_ids:
                infos.append([self.img_buffer[j], self.box_buffer[j]])
            img_path, box = self.mosaic(infos, self.imgsz)

            pass
        if self.epoch_now < self.stop_mosaic_epoch and self.config.mixup_prob>0:
            pass

    def load_image(self, i):
        img_path, box = self.imgs_path[i], self.boxs[i]
        img = read_img(img_path)
        img, flip = distor_image(img, self.imgsz)
        img = self.hsv(img)
        if flip and len(box)>0:
            box[:, [0,2]] = 1 - box[:, [2,0]]
        if len(self.buffer_idx) >= self.max_buffer_num:
            j = self.queue.popleft()
            self.img_buffer[j], self.label_buffer[j], self.box_buffer[j] = None, None, None
        self.buffer_idx.append(i)
        self.img_buffer[i], self.box_buffer[i] = img, box
        return img_path, box
            





    def __len__(self):
        return self.n_imgs

