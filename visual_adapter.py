from lime import lime_image, lime_text
import os
import cn_clip
import torch, tqdm, clip, time
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from mymodels import Adapter_Origin, Adapter_V1
from myconfig import Config
from cn_clip.clip import load_from_name
from skimage.segmentation import mark_boundaries

torch.manual_seed(0)
config = Config()
data_name = config.dataset_name
device = "cpu"
print(device)
adapter = Adapter_V1(num_classes=2).to(device)
adapter.load_state_dict(torch.load(config.save_path + "/adapter_shot16@gossipcop.pt"))

n_sample = 5000
if config.dataset_name == "weibo":
    model, preprocess = load_from_name("ViT-B-16", device=device)
else:
    model, preprocess = clip.load('ViT-B/32', device, jit=False)

class Pred_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, img_path):

        with open(data_path, 'r') as inf:
            self.data = pd.read_csv(inf, header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx][1] + ".jpg"
        label = self.data.iloc[idx][2]
        txt = self.data.iloc[idx][0]

        return txt, img, label

visual_samples = Pred_Dataset(config.pred_csv, config.pred_img_path)
txt, img, label = next(iter(visual_samples))

if config.dataset_name == "weibo":
    txt_prepro = cn_clip.clip.tokenize(txt).squeeze().to(device)
    img = preprocess(Image.open(config.pred_img_path + img)).to(device)
else:
    txt_prepro = clip.tokenize(txt, truncate=True).squeeze().to(device)
    img = preprocess(Image.open(config.pred_img_path + img)).to(device)

img = img.to(device).cpu().numpy() # (3, 224, 224)
img = np.transpose(img, (1, 2, 0))

def img_pred(im):
    with torch.no_grad():
        im = np.transpose(im, (0, 3, 1, 2))
        img_feat_1 = model.encode_image(torch.tensor(im))
        txt_feat_1 = model.encode_text(torch.tensor(txt_prepro).unsqueeze(0))
        txt_feat_1 = txt_feat_1.repeat(10, 1)

        img_feat = img_feat_1 / img_feat_1.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat_1 / txt_feat_1.norm(dim=-1, keepdim=True)

        all_feat = torch.cat((img_feat, txt_feat), dim=-1).to(device, torch.float32)
        adapter.eval()
        txt_logits, img_logits, eval_logits = adapter(txt_feat_1.to(device, torch.float32),
                              img_feat_1.to(device, torch.float32), all_feat)


        img_probs = torch.softmax(img_logits, dim=1)
        return img_probs.detach().cpu().numpy()

def txt_pred(txt):
    txt_prepro_ = clip.tokenize(txt, truncate=True).squeeze().to(device)
    with torch.no_grad():

        im = np.transpose(img, (2, 0, 1))

        img_feat_1 = model.encode_image(torch.tensor(im).unsqueeze(0))
        txt_feat_1 = model.encode_text(torch.tensor(txt_prepro_))
        img_feat_1 = img_feat_1.repeat(n_sample, 1)

        img_feat = img_feat_1 / img_feat_1.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat_1 / txt_feat_1.norm(dim=-1, keepdim=True)

        all_feat = torch.cat((img_feat, txt_feat), dim=-1).to(device, torch.float32)
        adapter.eval()
        txt_logits, img_logits, eval_logits = adapter(txt_feat_1.to(device, torch.float32),
                              img_feat_1.to(device, torch.float32), all_feat)


        txt_probs = torch.softmax(img_logits, dim=1)
        return txt_probs.detach().cpu().numpy()
print("start to explain image.....")
img_explainer = lime_image.LimeImageExplainer()
img_explanation = img_explainer.explain_instance(img,
                                         img_pred,
                                         labels=(0, 1),# classification function
                                         num_samples=n_sample,
                                        )

temp, mask = img_explanation.get_image_and_mask(img_explanation.top_labels[0], positive_only=False,
                                                negative_only=False, num_features=5, hide_rest=False,
                                                min_weight=0.)
img_boundry2 = mark_boundaries(temp, mask, color=(1,1,1), mode='inner')
plt.imshow(img_boundry2)



print("start to explain text.....")
txt_explainer = lime_text.LimeTextExplainer(class_names=["real", "fake"])
txt_explaination = txt_explainer.explain_instance(txt, txt_pred, num_features=5, num_samples=n_sample)

save_path = "./explain"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(save_path + "/img_explain.png")
txt_explaination.save_to_file(save_path +"/txt_explain.html")