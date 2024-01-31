# Cross-Modal Augmentation for Few-shot Multimodal Fake News Detection
This is the official implementation for the paper "Cross-Modal Augmentation for Few-shot Multimodal Fake News Detection".

# To run

1. `pip install -f requriements.txt`
2. prepares the `./datasets` dirs as follows:
```bash
├── datasets
│   ├── fakenewsnet
│   │   ├── goss_img_all
│   │	│	├── gossipcop-679264.jpg
│   │	│	├── gossipcop-681826.jpg
│   │	│       └── ....
│   │   ├── poli_img_all
│   │	│       ├── politifact66.jpg
│   │	│	├── politifact87.jpg
│   │   │       └── ....
│   │   ├── politifact_multi.csv
│   │   ├── gossipcop_multi.csv
│   ├── weibo
│   │   ├── all_images
│   │   │	├── 3b9c937bjw1e3rs645jowj.jpg
│   │   │	├── 3bb5c030jw1dzp3wwf3z3j.jpg
│   │   │       └── ....
│   │   ├── weibo_train.csv
│   │   ├── weibo_test.csv
```
3. Run `./run.sh`
