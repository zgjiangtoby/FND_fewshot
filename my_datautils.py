import os.path
from collections import defaultdict
import tqdm, random
from torch.utils.data import Dataset, Subset
import csv, clip, torch, cn_clip
import numpy as np
import pandas as pd
from PIL import Image
from cn_clip.clip import load_from_name
from myconfig import Config

## 1 fake, 0 real
config = Config()
class FakeNews_Dataset(Dataset):
    def __init__(self, data_path, img_path ):

        self.img_path = img_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.dataset_name == "weibo":
            model, self.preprocess = load_from_name("ViT-B-16", device=self.device)
        else:
            print("Loading OpenAI CLIP.....")
            model, self.preprocess = clip.load('ViT-B/32', self.device, jit=False)
        with open(data_path, 'r') as inf:
            self.data = pd.read_csv(inf)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx][1] + ".jpg"
        label = self.data.iloc[idx][2]
        txt = self.data.iloc[idx][0]
        if config.dataset_name == "weibo":
            txt = cn_clip.clip.tokenize(txt).squeeze().to(self.device)
            img = self.preprocess(Image.open(self.img_path + img)).to(self.device)
        else:
            txt = clip.tokenize(txt, truncate=True).squeeze().to(self.device)
            img = self.preprocess(Image.open(self.img_path + img)).to(self.device)
        label = torch.as_tensor(int(label)).to(self.device, torch.long)

        return txt, img, label


class FewShotSampler_weibo:
    def __init__(self, dataset, few_shot_per_class):
        self.dataset = dataset
        self.few_shot_per_class = few_shot_per_class

    def get_train_dataset(self):
        indices_per_class = defaultdict(list)
        for idx in range(len(self.dataset)):
            _, _, label = self.dataset[idx]
            indices_per_class[label.item()].append(idx)

        train_indices = []

        # 对于每个类别，随机选择few_shot_per_class个样本作为训练集
        for label, indices in indices_per_class.items():
            random.shuffle(indices)
            train_indices.extend(indices[:self.few_shot_per_class])

        train_dataset = Subset(self.dataset, train_indices)

        return train_dataset
class FewShotSampler_fakenewsnet:
    def __init__(self, dataset, few_shot_per_class):
        self.dataset = dataset
        self.few_shot_per_class = few_shot_per_class

    def get_train_val_datasets(self):
        indices_per_class = defaultdict(list)
        for idx in range(len(self.dataset)):
            _, _, label = self.dataset[idx]
            indices_per_class[label.item()].append(idx)

        train_indices = []
        val_indices = []

        # 对于每个类别，随机选择few_shot_per_class个样本作为训练集
        for label, indices in indices_per_class.items():
            random.shuffle(indices)
            train_indices.extend(indices[:self.few_shot_per_class])
            val_indices.extend(indices[self.few_shot_per_class:])

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)

        return train_dataset, val_dataset

def sim_cal(txt_path, img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device,
                                       download_root='~/PycharmProjects/local_models/clip-chinese')
    model.eval()
    cos = torch.nn.CosineSimilarity()
    all_txt, all_img, all_label = [], [], []
    with open(txt_path, 'r') as inf:
        content = inf.readlines()
        for i in tqdm.tqdm(range(0, len(content), 3)):
            # 获取连续的三行
            three_lines = content[i:i + 3]
            if len(three_lines[2].strip()) != 0:  # remove empty text
                urls = three_lines[1].split("|")
                txt = cn_clip.clip.tokenize(three_lines[2].strip()).to(device)
                txt_emb = model.encode_text(txt)
                img_final = None
                tmp_sim = 0
                for item in urls:
                    if item != 'null\n':
                        url = item.split("/")[-1]
                        if os.path.exists(img_path + url):
                            img = preprocess(Image.open(img_path + url)).unsqueeze(0).to(device)
                            img_emb = model.encode_image(img)
                            sim = cos(img_emb, txt_emb).cpu().detach().numpy()
                            if sim >= tmp_sim:
                                tmp_sim = sim
                                img_final = url
                if img_final is not None:
                    img_final = img_final.split(".")[0]
                    all_img.append(img_final)
                    all_txt.append(three_lines[2].strip())
                    if img_path.split("/")[-2] == "rumor_images":
                        all_label.append("1")
                    else:
                        all_label.append("0")

    return all_txt, all_img, all_label

def weibo_2_csv(fake_txt, fake_img, real_txt, real_img, out_path):
    ftxt, fimg, flabel = sim_cal(fake_txt, fake_img)
    rtxt, rimg, rlabel = sim_cal(real_txt, real_img)

    txt = ftxt + rtxt
    img = fimg + rimg
    label = flabel + rlabel
    with open(out_path, 'w', newline='') as outf:
        csv_writer = csv.writer(outf)
        for l in range(len(label)):
            csv_writer.writerow([txt[l], img[l], label[l]])

def load_from_csv(csv_path, img_path):
    all_txt, all_img, all_label = [], [], []
    with open(csv_path, 'r') as inf:
        data = csv.reader(inf)
        for line in data:
            img = line[2] + ".jpg"
            label = line[3]
            txt = line[1]
            all_img.append(img_path + img)
            all_txt.append(txt)
            all_label.append(label)

    return all_txt, all_img, all_label

# weibo_2_csv("../datasets/weibo/tweets/train_rumor.txt",
#             "../datasets/weibo/rumor_images/",
#              "../datasets/weibo/tweets/train_nonrumor.txt",
#             "../datasets/weibo/nonrumor_images/",
#             "../datasets/weibo/weibo_train.csv",
#             )


