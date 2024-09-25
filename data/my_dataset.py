from torch.utils.data.dataset import Dataset
from springc_utils import *
from collections import deque
from data.data_augment import *


def dataset_collate(batch):
    images  = []
    bboxes  = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes

class YOLODataset(Dataset):

    def __init__(self, img_info, config_yaml, class_name, imgsz) -> None:
        self.imgsz = imgsz
        self.imgs_path = img_info['img_info']
        #   boxs  
        #     (N, 5) 5: x1, y1, x2, y2, cls ,如果不符合格式，在输入YOLODataset之前处理/  
        #     (N, 6) 5: x1, y1, x2, y2, cls, angle 这里跟我做的一个项目有关，需要预测车辆角度，相当于多了一个属性
        self.boxs = img_info['boxs'] # 
        self.data_queue = []
        self.config = type('TrainingConfig', (object,), config_yaml) #把读取的yaml文件转成类，使用self.config.xxx 引用参数，避免每次使用字典方式引用
        self.n_class = len(class_name)
        self.class_name = class_name
        self.epoch_now = -1 #判断是否继续使用 mosaic/mixup 这类数据增强
        self.stop_mosaic_epoch = self.config.stop_mosaic_epoch
        self.n_imgs = len(self.imgs_path)
        self.img_buffer, self.box_buffer = [None] * self.n_imgs, [None] * self.n_imgs
        self.buffer_idx = deque()
        self.max_buffer_num = self.config.max_buffer_num
        self.hsv = RandomHSV()
        self.mosaic = Mosaic()
        
    def preprocess_input(self, image):
        image = image[:, :, ::-1]
        image /= 255.0
        return image


    def __getitem__(self, i):
        img, box = self.load_image(i)
        if self.epoch_now < self.stop_mosaic_epoch and self.config.mosaic_prob>0:
            selected_ids = random.choice(self.buffer_idx, k=3)
            selected_ids.append(i)
            infos = []
            for j in selected_ids:
                infos.append([self.img_buffer[j], self.box_buffer[j]])
            img, box = self.mosaic(infos, self.imgsz)
        if self.epoch_now < self.stop_mosaic_epoch and self.config.mixup_prob>0:
            pass
        img = np.transpose(self.preprocess_input(np.array(img, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        n, c = box.shape
        labels_out  = np.zeros((n, c+1))
        if n:
            labels_out[:, 1:] = box # 首位的索引，用来后面标记这是第几个图像索引 img_cnt, x1, y1, x2, y2, cls, (angle，如果有的话)
        return img, labels_out
        
        
        

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
        return img, box
            


    def __len__(self):
        return self.n_imgs

