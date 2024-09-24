from albumentations import Compose, HueSaturationValue
import cv2
import numpy as np
from PIL import Image

def sc_rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def distor_image(image, size, jitter):
    """
    扭曲图像
    image: cv2读取的图像
    size: 640
    jitter: 扭曲参数
    """
    if isinstance(size, tuple) or isinstance(size, list):
        size = size[0] 
    iw, ih = image.size
    flip = np.random.rand()<0.5
    if flip:
        image = cv2.flip(image, 1) # 1代表水平翻转
    #   对图像进行缩放并且进行长和宽的扭曲
    new_ar = iw/ih * sc_rand(1-jitter,1+jitter) / sc_rand(1-jitter,1+jitter)
    scale = sc_rand(0.4, 1)
    if new_ar < 1:
        nh = int(scale * size)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * size)
        nh = int(nw / new_ar)
    interp = cv2.INTER_LINEAR if max(nw, nh)> max(iw, ih) else cv2.INTER_AREA
    image = cv2.resize(image, (nw, nh), interpolation=interp)
    return image, flip
class RandomHSV:

    """
    HSV 色域变换
    hgain（色相增益）：
        影响图像的色相（Hue），即颜色的基本属性（如红色、绿色等）。
        增益值范围为 -1 到 1，正值会使色相偏向其原本的颜色，负值则可能导致颜色的变化，例如将红色偏向蓝色。

    sgain（饱和度增益）：
        控制图像的饱和度（Saturation），即颜色的鲜艳程度。
        增益值大于 1 会增加饱和度，使颜色更鲜艳；小于 1 则会降低饱和度，使颜色更接近灰色。

    vgain（明度增益）：
        影响图像的明度（Value），即亮度的程度。
        增益值大于 1 会使图像变亮，小于 1 则会使图像变暗。
    h, s, v=1时，相当于不做任何色域变换
    """
    
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img):
        """Applies random horizontal or vertical flip to an image with a given probability."""
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BRG2HSV))
        dtype = img.dtype  # uint8
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BRG)  # no return needed
        return img


    




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

        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)




            