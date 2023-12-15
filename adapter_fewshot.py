import torch, tqdm, clip, time
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from my_datautils import FakeNews_Dataset, FewShotSampler_fakenewsnet, FewShotSampler_weibo
from mymodels import Adapter_Origin, Adapter_V1
from myconfig import Config
from cn_clip.clip import load_from_name

torch.manual_seed(0)
config = Config()
data_name = config.dataset_name

# weibo: adapter_origin: 0.617@2shots; 0.722@16shots; 0.797@100shots
# goss: adapter_origin: 0.612@2shots; 0,567@16shots; adapter_v1: 0.673@16shots
# poli:

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
if data_name == "weibo":
    print("Loading Chinese CLIP.....")
    model, preprocess = load_from_name("ViT-B-16", device=device)
    train_dataset = FakeNews_Dataset(config.train_csv, config.img_path)
    test_dataset = FakeNews_Dataset(config.test_csv, config.img_path)

    train_sampler = FewShotSampler_weibo(train_dataset, config.few_shot_per_class)
    train_dataset = train_sampler.get_train_dataset()

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)

else:
    print("Loading OpenAI CLIP.....")
    model, preprocess = clip.load('ViT-B/32', device, jit=False)
    train_dataset = FakeNews_Dataset(config.train_csv, config.img_path)

    sampler = FewShotSampler_fakenewsnet(train_dataset, config.few_shot_per_class)

    # 分割数据集
    train_dataset, test_dataset = sampler.get_train_val_datasets()
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)


# adapter = Adapter_Origin(num_classes=2).to(device)
adapter = Adapter_V1(num_classes=2).to(device)
EPOCH = 20
optimizer = AdamW(adapter.parameters(), lr=1e-3, eps=1e-4)
# scheduler = CosineAnnealingLR(optimizer, EPOCH * len(train_loader))
loss_func = CrossEntropyLoss()
best_test_acc_in_epoch, patience_count, patience = 0, 0, 3

for epoch in range(EPOCH):

    print("EPOCH: {} ".format(epoch + 1))
    for txt, img, label in tqdm.tqdm(train_loader):
        adapter.train()
        img_feat_0 = model.encode_image(img)
        txt_feat_0 = model.encode_text(txt)

        # label = label
        img_feat = img_feat_0 / img_feat_0.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat_0 / txt_feat_0.norm(dim=-1, keepdim=True)

        all_feat = torch.cat((img_feat, txt_feat), dim=-1).to(device, torch.float32)

        _, _ ,logits = adapter(txt_feat_0.to(device, torch.float32),
                               img_feat_0.to(device, torch.float32), all_feat)
        loss = loss_func(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        end_time = time.time()


    test_labels = []
    pred_labels = []

    print("Start to eval: ")
    with torch.no_grad():
        for txt, img, label in tqdm.tqdm(test_loader):
            img_feat_1 = model.encode_image(img)
            txt_feat_1 = model.encode_text(txt)

            img_feat = img_feat_1 / img_feat_1.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat_1 / txt_feat_1.norm(dim=-1, keepdim=True)

            all_feat = torch.cat((img_feat, txt_feat), dim=-1).to(device, torch.float32)
            adapter.eval()
            _, _, eval_logits = adapter(txt_feat_1.to(device, torch.float32),
                                  img_feat_1.to(device, torch.float32), all_feat)

            predictions = torch.argmax(torch.softmax(eval_logits, dim=1), dim=-1).cpu().numpy()
            label = label.cpu().numpy()
            accuracy = np.mean((label == predictions).astype(float)) * 100.

            test_labels.extend(label)
            pred_labels.extend(predictions)

    report = classification_report(test_labels, pred_labels, output_dict=True)
    print(f"Accuracy = {report['accuracy']:.3f}")
    epoch_acc = round(report['accuracy'], 3)
    if epoch_acc > best_test_acc_in_epoch:
        best_test_acc_in_epoch = epoch_acc
        print("Save best acc at EPOCH {}".format(epoch + 1))
        patience_count = 0
    else:
        patience_count += 1

    if patience_count >= patience:
        print("Early stopping triggered")
        break  # 跳出训练循环

print("best_acc found at: ", best_test_acc_in_epoch)