目标step：
- 自己的理解复现yolov8
    - 参考ultra数据加载方式，使用队列来加快mosaic数据增强
- 替换损失函数为yolox
- 替换模型为yolox
- 类似插件方式，自定义选择模型和损失函数:yolov8还是yolox

数据格式为：
一个jpg对应一个txt文件
txt文件中每一行代表一个目标
每个目标： cls, cx, cy, w, h 且归一化为0-1

训练时，给定一个txt文件，txt文件每一行代表一个jpg_file,替换掉后缀名为txt后，为txt路径
input_shape: (h, w)