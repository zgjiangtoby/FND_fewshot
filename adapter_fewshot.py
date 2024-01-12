import os.path
import argparse
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

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="help")
        parser.add_argument("--seed", type=int, help="this is seed number")
        parser.add_argument("--dataset_name", type=str, help="weibo, politifact, gossipcop")
        parser.add_argument("--train_csv", type=str, help="this is your train directory")
        parser.add_argument("--test_csv", type=str, help="this is your test directory")
        parser.add_argument("--img_path", type=str, help="img path")
        parser.add_argument("--shot", type=int, help='shots')
        parser.add_argument("--save_path", type=str, help="train model")


        args = parser.parse_args()

        set_seeds(args.seed)

        data_name = args.dataset_name

        seed = args.seed
        print("SEED : ", seed)

        print(device)
        if data_name == "weibo":
            print("Loading Chinese CLIP.....")
            model, preprocess = load_from_name("ViT-B-16", device=device)
            train_dataset = FakeNews_Dataset(model, preprocess, args.train_csv, args.img_path, data_name)
            test_dataset = FakeNews_Dataset(model, preprocess, args.test_csv, args.img_path, data_name)

            train_sampler = FewShotSampler_weibo(train_dataset, args.shot, seed)
            train_dataset = train_sampler.get_train_dataset()
            torch.manual_seed(args.seed)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=lambda seed: np.random.seed(seed))
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        else:
            print("Loading OpenAI CLIP.....")
            model, preprocess = clip.load('ViT-B/32', device, jit=False)
            train_dataset = FakeNews_Dataset(model, preprocess, args.train_csv, args.img_path, data_name)

            sampler = FewShotSampler_fakenewsnet(train_dataset, args.shot, seed)

            # 分割数据集
            train_dataset, test_dataset = sampler.get_train_val_datasets()
            # 创建 DataLoader
            torch.manual_seed(args.seed)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=lambda seed: np.random.seed(seed))
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


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

                _, _, logits = adapter(txt_feat_0.to(device, torch.float32),
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
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                print("saving best model at Epoch {} | Accuracy {}".format(epoch+1, epoch_acc))
                torch.save(adapter.state_dict(),
                           args.save_path + "/adapter_shot{}@{}.pt".format(args.shot, args.dataset_name))
            else:
                patience_count += 1

            if patience_count >= patience:
                print("Early stopping triggered")
                break  # 跳出训练循环

        print("best_acc found at: ", best_test_acc_in_epoch)
        with open(args.save_path + "/seed{}_shot{}@{}.txt".format(args.seed, args.shot, args.dataset_name), 'w') as outf:
            outf.write(str(round(best_test_acc_in_epoch, 4)))