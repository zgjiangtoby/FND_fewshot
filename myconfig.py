
class Config:

    dataset_name = "gossipcop"

    if dataset_name == "weibo":
        train_csv = "datasets/weibo/weibo_train.csv"
        test_csv = "datasets/weibo/weibo_test.csv"
        img_path = "datasets/weibo/all_images/"

    elif dataset_name == "polifact":
        train_csv = "datasets/fakenewsnet/polifact_multi.csv"
        img_path = "datasets/fakenewsnet/poli_img_all/"
    elif dataset_name == "gossipcop":
        train_csv = "datasets/fakenewsnet/gossipcop_multi.csv"
        img_path = "datasets/fakenewsnet/goss_img_all/"

    pred_csv = "datasets/fakenewsnet/test_fake.csv"
    pred_img_path = "datasets/fakenewsnet/poli_img_all/"

    save_path = "./saved_adapter"
    version = 1
    few_shot_per_class = 100