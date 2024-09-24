from springc_utils import *
from tqdm import tqdm
import yaml
setup_logging()

def get_txtinfo_xywh(txt_path):
    cache_path = txt_path[:-3] + "cache.npy"
    if os.path.exists(cache_path):
        res_dict = np.load(cache_path, allow_pickle=True).item()
        return res_dict
    res_dict = {}
    res_dict["img_path"] = []
    res_dict["labels"] = []
    res_dict["boxs"] = []
    suffixes = (".jpg", ".png")
    cnt = 0
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().splitlines()
        for line in tqdm(lines):
            cnt +=1
            # if cnt>200:break
            line = line.strip()
            img = read_img(line)
            if img is None:
                logging.info(f"图像不存在：{line}")
            assert line.endswith(suffixes), logging.info(f"图像不是以{suffixes}为后缀：{line}")
            label_path = line[:-3] + "txt"
            with open(label_path, 'r', encoding='utf-8') as tf:
                tlines = tf.read().strip().splitlines()
                if len(tlines)<1:
                    logging.info(f"图像的标注文件没有任何信息：{line}")
                tlines = [t.split() for t in tlines]
                tlines = np.asarray(tlines).astype(np.float32)
                tlabel = tlines[:, 0]
                tbox = tlines[:, 1:]
                res_dict["labels"].append(tlabel)
                res_dict["boxs"].append(tbox)
                res_dict["img_path"].append(line)
    np.save(cache_path, res_dict)
    return res_dict
                

def read_yaml(file):
    with open(file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

p1 = r"/mnt/hd1/springc/code/github/YoloV8/config/data.yaml"
config = read_yaml(p1)
config['names'].values()