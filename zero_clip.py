import clip, torch, os, csv, tqdm
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from sklearn.metrics import classification_report
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, KFold


torch.manual_seed(0)

all_txt = []
all_img = []
all_label = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device, jit=False, download_root='~/PycharmProjects/local_models/clip_origin')
print(device)
with open("/home/yejiang/PycharmProjects/prompt-fakenews/FakeNewsNet-master/politifact_multi.csv", 'r') as inf:
    data = csv.reader(inf)
    for line in data:
        img = line[2] + ".jpg"
        label = line[3]
        txt = line[1]
        all_img.append("/home/yejiang/PycharmProjects/prompt-fakenews/FakeNewsNet-master/code/fakenewsnet_dataset/politifact_multi/poli_img_all/"+ img)
        all_txt.append(txt)
        all_label.append(label)

n_fold = 5
K_fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=2025)


class Example_Dataset(Dataset):
    def __init__(self, img, txt, label, preprocessor):
        self.img = img
        self.txt = txt
        self.label = label
        self.preprocessor = preprocessor
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        txt = clip.tokenize(self.txt[idx], truncate=True).squeeze().to(device)
        img = self.preprocessor(Image.open(self.img[idx])).to(device)
        label = torch.as_tensor(int(self.label[idx])).to(device, torch.float32)

        return txt, img, label

dataset = Example_Dataset(all_img, all_txt, all_label, preprocess)

class MyModel(torch.nn.Module):
    def __init__(self, clip_model, num_classes=2):
        super(MyModel, self).__init__()
        self.clip_model = clip_model
        self.fc = torch.nn.Linear(512*2, num_classes, bias=False).half()  # 假设clip_model.output_dim是CLIP输出的维度

        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, txt, img):

        img_feat = model.encode_image(img)
        txt_feat = model.encode_text(txt)

        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

        all_feat = torch.cat((img_feat, txt_feat), dim=-1).half()
        x = self.fc(all_feat)
        return torch.softmax(x, dim=1)


batch_size = 8
train_dataloader = DataLoader(dataset, batch_size, shuffle=True)

w = MyModel(model, num_classes=2).to(device)
steps = 1
all_acc = 0

test_labels = []
pred_labels = []

for txt, img, label in tqdm.tqdm(train_dataloader):
    with torch.no_grad():
        logit = w(txt, img)
        predictions = torch.argmax(logit, dim=-1).cpu().numpy()
        label = label.cpu().numpy()
        accuracy = np.mean((label == predictions).astype(float)) * 100.

        all_acc += accuracy
        steps += 1

        test_labels.extend(label)
        pred_labels.extend(predictions)

print(f"Accuracy = {all_acc/steps:.3f}")
r = classification_report(test_labels, pred_labels, output_dict=True)
print(r)
print(f"Accuracy = {r['accuracy']:.3f}")

# for txt, img, label in tqdm.tqdm(val_dataloader):
#     with torch.no_grad():
#         img_feat = model.encode_image(img)
#
#         txt_feat = model.encode_text(txt)
#
#         img_feat /= img_feat.norm(dim=-1, keepdim=True)
#         txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
#
#         if multi:
#             all_feat = torch.cat((img_feat, txt_feat), dim=-1)
#         else:
#             all_feat = txt_feat
#         val_feature.append(all_feat)
#         val_labels.append(label)
#
# train_feature = torch.cat(train_feature).cpu().numpy()
# train_labels = torch.cat(train_labels).cpu().numpy()
#
# val_feature = torch.cat(val_feature).cpu().numpy()
# val_labels = torch.cat(val_labels).cpu().numpy()
#
# classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
# classifier.fit(train_feature, train_labels)
#
# predictions = classifier.predict(val_feature)
# accuracy = np.mean((val_labels == predictions).astype(float)) * 100.
# print(f"Accuracy = {accuracy:.3f}")
