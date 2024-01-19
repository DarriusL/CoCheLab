# CoCheLab

Code for the **Co**ntent Ca**ch**ing algorithm in edge caching.

## Environment configuration

```shell
git clone https://github.com/DarriusL/CacheLab.git
```

Create an environment using one of these methods:

### conda

```shell
cd CacheLab
conda env create -f cachelab_dev.yml
conda activate cachelab_dev
```

### pip

```shell
conda create -n CacheLab python=3.11
pip install -r requirements.txt
```



## Framework file structure

```
├── .gitignore
├── cache
├── cachelab_env.yml
├── config
│	├── caser
│	│	├── caser_appliances.json
│	│	├── caser_ml1m.json
│	│	└── caser_music.json
│	├── CFG_README.md
│	├── cl4srec
│	│	├── cl4srec_appliances.json
│	│	├── cl4srec_ml1m.json
│	│	└── cl4srec_music.json
│	├── data_process_cfg.json
│	├── duo4srec
│	│	├── duo4srec_appliances.json
│	│	├── duo4srec_ml1m.json
│	│	└── duo4srec_music.json
│	├── ec4srec
│	│	├── ec4srec_appliances.json
│	│	├── ec4srec_ml1m.json
│	│	└── ec4srec_music.json
│	├── egpc
│	│	├── egpc_appliances.json
│	│	├── egpc_ml1m.json
│	│	└── egpc_music.json
│	├── fifo
│	│	├── fifo_appliances.json
│	│	├── fifo_ml1m.json
│	│	└── fifo_music.json
│	├── lab_cfg.json
│	├── lfu
│	│	├── lfu_appliances.json
│	│	├── lfu_ml1m.json
│	│	└── lfu_music.json
│	├── lru
│	│	├── lru_appliances.json
│	│	├── lru_ml1m.json
│	│	└── lru_music.json
│	└── psac
│		├── psac_gen_appliances.json
│		├── psac_gen_ml1m.json
│		└── psac_gen_music.json
├── data
│	├── augmentation.py
│	├── datasets
│	│	├── meta
│	│	│	├── appliances
│	│	│	│	├── Appliances.json
│	│	│	│	└── Appliances_lite.json
│	│	│	├── beauty
│	│	│	│	├── All_Beauty.json
│	│	│	│	└── All_Beauty_lite.json
│	│	│	├── kindle
│	│	│	│	├── Kindle_Store.json
│	│	│	│	└── Kindle_Store_lite.json
│	│	│	├── ml-1m
│	│	│	│	├── movies.dat
│	│	│	│	├── ratings.dat
│	│	│	│	├── README
│	│	│	│	└── users.dat
│	│	│	└── music
│	│	│		├── Digital_Music.json
│	│	│		└── Digital_Music_lite.json
│	│	└── process
│	│		├── complete
│	│		│	├── All_Beauty.data
│	│		│	├── Appliances.data
│	│		│	├── appliances_devide_25.data
│	│		│	├── Digital_Music.data
│	│		│	├── kindle_devide_25.data
│	│		│	├── Kindle_Store.data
│	│		│	├── ml.data
│	│		│	├── ml_devide_55.data
│	│		│	└── music_devide_25.data
│	│		└── lite
│	│			├── All_Beauty_lite.data
│	│			├── Appliances_lite.data
│	│			├── Digital_Music_lite.data
│	│			├── Kindle_Store_lite.data
│	│			├── ml.data
│	│			└── music_devide_25.data
│	├── generator.py
│	├── processor.py
│	├── saved./
│	└── __init__.py
├── executor.py
├── lib
│	├── callback.py
│	├── glb_var.py
│	├── json_util.py
│	└── util.py
├── LICENSE
├── model
│	├── attnet.py
│	├── cnnnet.py
│	├── framework
│	│	├── base.py
│	│	├── caser.py
│	│	├── cl4srec.py
│	│	├── duo4srec.py
│	│	├── ec4srec.py
│	│	├── egpc.py
│	│	├── fifo.py
│	│	├── lfu.py
│	│	├── lru.py
│	│	└── psac.py
│	├── loss.py
│	└── __init__.py
├── README.md
├── requirements.txt
└── Room
	├── officer.py
	├── work.py
	└── __init__.py

```



**Dataset acquisition:**

---

[ML-1M](https://grouplens.org/datasets/movielens/1m/)

[Amazon review lite](https://nijianmo.github.io/amazon/)

[Amazon review](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) 

The original dataset is downloaded and saved in the path:  ./data/datasets/meta/

At the same time, I also provide the google drive link to the [original data set](https://drive.google.com/drive/folders/1W8oUXWaX_rn90s0R78gLNvhQOBgIRgN9?usp=drive_link) and the [processed data set](https://drive.google.com/drive/folders/1dBZQZtmSbVeCLOlM0SimZimaBG0JFWqF?usp=drive_link).

If you need to use the provided processed data set, download it to path : ./data/dataset/process/

**Algorithms that have been implemented and algorithms that will be supported soon:**

----------------------------------------------------

Coventional:

- [x] FIFO
- [x] LRU
- [x] LFU

CL-based: 

- [x] CL4SRec[[1]](#ref1)	
- [x] Duo4SRec(DuoSRec)[[2]](#ref2)
- [x] EC4SRec[[3]](#ref3)
- [x] EGPC

DL-based:

- [x] PSAC_gen[[4]](#ref4)
- [x] Caser[[5]](#ref5)



**Configuration file related**: [CFG_README.md](./config/CFG_README.md)



## Command

### usage

```shell
usage: executor.py [-h] [--data_process DATA_PROCESS] [--config CONFIG] [--saved_config SAVED_CONFIG] [--mode MODE]
```

### options

```shell
-h, --help            show this help message and exit
--data_process DATA_PROCESS, -dp DATA_PROCESS
                    type for data process(None/lite/complete)
--config CONFIG, -cfg CONFIG
                    config for run
--saved_config SAVED_CONFIG, -sc SAVED_CONFIG
                    path for saved config to test
--mode MODE           train/test/train_and_test
```

For configuration files, see:./config/CFG_README.md

### quick start

process dataset

You need to configure the data processing configuration file yourself:@./config/data_process_cfg.json

```
python executor.py --data_process=True
```

FIFO

```shell
python executor.py -sc='./config/fifo/fifo_ml1m.json' --mode=test
python executor.py -sc='./config/fifo/fifo_appliances.json' --mode=test
python executor.py -sc='./config/fifo/fifo_music.json' --mode=test
```

LRU

```shell
python executor.py -sc='./config/lru/lru_ml1m.json' --mode=test
python executor.py -sc='./config/lru/lru_appliances.json' --mode=test
python executor.py -sc='./config/lru/lru_music.json' --mode=test
```

LFU

```shell
python executor.py -sc='./config/lfu/lfu_ml1m.json' --mode=test
python executor.py -sc='./config/lfu/lfu_appliances.json' --mode=test
python executor.py -sc='./config/lfu/lfu_music.json' --mode=test
```

CL4SRec

```shell
python executor.py -cfg='./config/cl4srec/cl4srec_ml1m.json' --mode=train
python executor.py -cfg='./config/cl4srec/cl4srec_ml1m.json' --mode=train_and_test
python executor.py -sc='./data/saved/cl4srec/ml1m/pre_64_2048_2_2/config.json' --mode=test

python executor.py -cfg='./config/cl4srec/cl4srec_appliances.json' --mode=train
python executor.py -cfg='./config/cl4srec/cl4srec_appliances.json' --mode=train_and_test
python executor.py -sc='./data/saved/cl4srec/appliances/pre_64_2048_2_2/config.json' --mode=test

python executor.py -cfg='./config/cl4srec/cl4srec_music.json' --mode=train
python executor.py -cfg='./config/cl4srec/cl4srec_music.json' --mode=train_and_test
python executor.py -sc='./data/saved/cl4srec/music/pre_64_2048_2_2/config.json' --mode=test
```

Duo4SRec

```shell
python executor.py -cfg='./config/duo4srec/duo4srec_ml1m.json' --mode=train
python executor.py -cfg='./config/duo4srec/duo4srec_ml1m.json' --mode=train_and_test
python executor.py -sc='./data/saved/duo4srec/ml1m/pre_64_2048_2_2/config.json' --mode=test

python executor.py -cfg='./config/duo4srec/duo4srec_appliances.json' --mode=train
python executor.py -cfg='./config/duo4srec/duo4srec_appliances.json' --mode=train_and_test
python executor.py -sc='./data/saved/duo4srec/appliances/pre_64_2048_2_2/config.json' --mode=test

python executor.py -cfg='./config/duo4srec/duo4srec_music.json' --mode=train
python executor.py -cfg='./config/duo4srec/duo4srec_music.json' --mode=train_and_test
python executor.py -sc='./data/saved/duo4srec/music/pre_64_2048_2_2/config.json' --mode=test
```

EC4SRec

```shell
python executor.py -cfg='./config/ec4srec/ec4srec_ml1m.json' --mode=train
python executor.py -cfg='./config/ec4srec/ec4srec_ml1m.json' --mode=train_and_test
python executor.py -sc='./data/saved/ec4srec/ml1m/pre_64_2048_2_2/config.json' --mode=test

python executor.py -cfg='./config/ec4srec/ec4srec_appliances.json' --mode=train
python executor.py -cfg='./config/ec4srec/ec4srec_appliances.json' --mode=train_and_test
python executor.py -sc='./data/saved/ec4srec/appliances/pre_64_2048_2_2/config.json' --mode=test

python executor.py -cfg='./config/ec4srec/ec4srec_music.json' --mode=train
python executor.py -cfg='./config/ec4srec/ec4srec_music.json' --mode=train_and_test
python executor.py -sc='./data/saved/ec4srec/music/pre_64_2048_2_2/config.json' --mode=test
```

EGPC

```shell
python executor.py -cfg='./config/egpc/egpc_ml1m.json' --mode=train
python executor.py -cfg='./config/egpc/egpc_ml1m.json' --mode=train_and_test
python executor.py -sc='./data/saved/egpc/ml1m/pre_64_2048_2_2/config.json' --mode=test

python executor.py -cfg='./config/egpc/egpc_appliances.json' --mode=train
python executor.py -cfg='./config/egpc/egpc_appliances.json' --mode=train_and_test
python executor.py -sc='./data/saved/egpc/appliances/pre_64_2048_2_2/config.json' --mode=test

python executor.py -cfg='./config/egpc/egpc_music.json' --mode=train
python executor.py -cfg='./config/egpc/egpc_music.json' --mode=train_and_test
python executor.py -sc='./data/saved/egpc/music/pre_64_2048_2_2/config.json' --mode=test
```

Caser

```shell
python executor.py -cfg='./config/caser/caser_ml1m.json' --mode=train
python executor.py -cfg='./config/caser/caser_ml1m.json' --mode=train_and_test
python executor.py -sc='./data/saved/caser/ml1m/64_8/config.json' --mode=test

python executor.py -cfg='./config/caser/caser_appliances.json' --mode=train
python executor.py -cfg='./config/caser/caser_appliances.json' --mode=train_and_test
python executor.py -sc='./data/saved/caser/appliances/64_8/config.json' --mode=test

python executor.py -cfg='./config/caser/caser_music.json' --mode=train
python executor.py -cfg='./config/caser/caser_music.json' --mode=train_and_test
python executor.py -sc='./data/saved/caser/music/64_8/config.json' --mode=test
```

PSAC

```shell
python executor.py -cfg='./config/psac/psac_gen_ml1m.json' --mode=train
python executor.py -cfg='./config/psac/psac_gen_ml1m.json' --mode=train_and_test
python executor.py -sc='./data/saved/psac_gen/ml1m/64_8/config.json' --mode=test

python executor.py -cfg='./config/psac/psac_gen_appliances.json' --mode=train
python executor.py -cfg='./config/psac/psac_gen_appliances.json' --mode=train_and_test
python executor.py -sc='./data/saved/psac_gen/appliances/64_8/config.json' --mode=test

python executor.py -cfg='./config/psac/psac_gen_music.json' --mode=train
python executor.py -cfg='./config/psac/psac_gen_music.json' --mode=train_and_test
python executor.py -sc='./data/saved/psac_gen/music/64_8/config.json' --mode=test
```



# Refrence

1. <spin id='ref1'></spin>Xie X, Sun F, Liu Z, et al. Contrastive learning for sequential recommendation[C]//2022 IEEE 38th international conference on data engineering (ICDE). IEEE, 2022: 1259-1273.
1. <spin id='ref2'></spin>Qiu R, Huang Z, Yin H, et al. Contrastive learning for representation degeneration problem in sequential recommendation[C]//Proceedings of the fifteenth ACM international conference on web search and data mining. 2022: 813-823.
1. <spin id='ref3'></spin>Wang L, Lim E P, Liu Z, et al. Explanation guided contrastive learning for sequential recommendation[C]//Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022: 2017-2027.
1. <spin id='ref4'></spin>Zhang Y, Li Y, Wang R, et al. PSAC: Proactive sequence-aware content caching via deep learning at the network edge[J]. IEEE Transactions on Network Science and Engineering, 2020, 7(4): 2145-2154.
1. <spin id='ref5'></spin>Tang J, Wang K. Personalized top-n sequential recommendation via convolutional sequence embedding[C]//Proceedings of the eleventh ACM international conference on web search and data mining. 2018: 565-573.

