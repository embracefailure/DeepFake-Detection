import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
from __init__ import get_model  # 确保这个模块存在并正确导入

SEED = 0

def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}

def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)

def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)

def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out

def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class TestDataset(Dataset):
    def __init__(self, test_path, arch, jpeg_quality=None, gaussian_sigma=None):
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        
        self.image_list = get_list(test_path)
        
        # Assign labels as None since we don't have real or fake labels for testing
        self.labels_dict = {i: None for i in self.image_list}

        stat_from = "imagenet" 
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img = Image.open(img_path).convert("RGB")
        
        print(img_path)

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, img_path  # Return image and its path for later use

def load_model(arch, ckpt):
    model = get_model(arch)
    state_dict = torch.load(ckpt, map_location='cpu')
    fc_state_dict = state_dict['model']
    with torch.no_grad():
        model.fc.weight.copy_(fc_state_dict['fc.weight'])
        model.fc.bias.copy_(fc_state_dict['fc.bias'])
    model.eval()
    model.cuda()
    return model

def predict_and_save_results(model, dataset, result_folder, batch_size=512):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    results = []
    
    with torch.no_grad():
        for img, img_path in tqdm(loader):
            in_tens = img.cuda()
            y_pred = model(in_tens).sigmoid().flatten().tolist()
            # 将预测结果转换为二进制
            y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
            results.extend(zip(img_path, y_pred_binary))  # 存储图片路径和预测

    # 保存结果到CSV文件
    with open(os.path.join(result_folder, 'cla_pre.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for filename, prediction in results:
            # 只取文件名，不取后缀
            name_without_ext = os.path.splitext(os.path.basename(filename))[0]
            writer.writerow([name_without_ext, prediction])

def main():
    test_path = '/root/testdata'  # 替换为测试集图片路径
    arch = 'DINO:ViT-L/14'  # 模型架构
    ckpt = './checkpoints/model.pth'  # 权重路径
    result_folder = 'result'  # 结果保存路径
    jpeg_quality = None  # 如果需要调整JPEG质量可以修改
    gaussian_sigma = None  # 如果需要高斯模糊可以修改

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    set_seed()
    
    dataset = TestDataset(test_path, arch, jpeg_quality, gaussian_sigma)
    model = load_model(arch, ckpt)
    
    predict_and_save_results(model, dataset, result_folder)

if __name__ == '__main__':
    main()