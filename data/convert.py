from albumentations import Compose, HueSaturationValue
import cv2
import numpy as np
from PIL import Image
import logging

class Mosaic():

    def __init__(self, min_side=3) -> None:
        self.min_side=3

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def __call__(self, infos, imgsz):
        self.mosaic4(infos, imgsz)
    
    def mosaic4(self, infos, imgsz):
        assert len(infos) == 4
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        midx, midy = int(imgsz*min_offset_x), int(imgsz*min_offset_y)
        image_datas = []
        box_datas = []
        for index, info in enumerate(infos):
            image, box = info # box:x1 y1 x2 y2, cls
            nh, nw, c = image.shape
            """
            | 0 | 3 |
            ---------
            | 1 | 2 |
            dx dy代表图片放入的左上角位置
            """
            if index == 0:
                dx = midx - nw
                dy = midy - nh
            elif index == 1:
                dx = midx - nw
                dy = midy
            elif index == 2:
                dx = midx
                dy = midy
            elif index == 3:
                dx = midx
                dy = midy - nh
            image = Image.fromarray(image)
            
            new_image = Image.new('RGB', (imgsz,imgsz), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)
            if len(box)>0:
                box[:, [0,2]] = box[:, [0,2]] * nw + dx
                box[:, [1,3]] = box[:, [1,3]] * nh + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2:4][box[:, 2:4]>imgsz] = imgsz
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>self.min_side, box_h>self.min_side)]# 任意边长小于min_side个像素的框废弃，太小了
                box_data = np.zeros(box.shape)
                box_data[:len(box)] = box
            image_datas.append(image_data)
            box_datas.append(box_data)

        new_image = np.zeros([imgsz, imgsz, 3])
        new_image[:midy, :midx, :] = image_datas[0][:midy, :midx, :]
        new_image[midy:, :midx, :] = image_datas[1][midy:, :midx, :]
        new_image[midy:, midx:, :] = image_datas[2][midy:, midx:, :]
        new_image[:midy, midx:, :] = image_datas[3][:midy, midx:, :]
        box_datas[:, :4] = self.merge_bboxes(box_datas[:, :4], midx, midy) / imgsz #归一化到0-1 x1 y1 x2 y2 
        return new_image, box_datas, 
        
        
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        logging.debug(f"出现异常：merge box in {i}")
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        logging.debug(f"出现异常：merge box in {i}")
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        logging.debug(f"出现异常：merge box in {i}")
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        logging.debug(f"出现异常：merge box in {i}")
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox




            