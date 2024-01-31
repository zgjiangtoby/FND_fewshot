# Cross-Modal Augmentation for Few-shot Multimodal Fake News Detection
This is the official implementation for the paper "Cross-Modal Augmentation for Few-shot Multimodal Fake News Detection".

# To run

1. `pip install -f requriements.txt`
2. prepares the dir `./datasets` as follow:
```bash
├── datasets
│   ├── fakenewsnet
│   │   ├── goss_img_all
│   │	│	├── gossipcop-679264.jpg
│   │	│	├── gossipcop-681826.jpg
│   │	│	│ 
│   │	└── ....
|   │   ├── poli_img_all
│   └── real
│      ├── gossipcop-1
│      │	├── news content.json
│      │	├── tweets
│      │	└── retweets
│		└── ....		
├── politifact
│   ├── fake
│   │   ├── politifact-1
│   │   │	├── news content.json
│   │   │	├── tweets
│   │   │	└── retweets
│   │	└── ....		
│   │
│   └── real
│      ├── poliifact-2
│      │	├── news content.json
│      │	├── tweets
│      │	└── retweets
│      └── ....					
├── user_profiles
│		├── 374136824.json
│		├── 937649414600101889.json
│   		└── ....
├── user_timeline_tweets
│		├── 374136824.json
│		├── 937649414600101889.json
│	   	└── ....
└── user_followers
│		├── 374136824.json
│		├── 937649414600101889.json
│	   	└── ....
└──user_following
        	├── 374136824.json
		├── 937649414600101889.json
	   	└── ....
```
