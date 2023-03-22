# CacheLab

Code for the edge cache algorithm.

## Environment configuration

```shell
cd CacheLab
conda env create -f environment.yml
conda activate cachelab
```

## Framework file structure

```
CacheLab
├── cache
│	├── logger
│	│	└── logger.log
│	└── unsaved_data
├── config
│	├── cl4srec
│	│	└── cl4srec_ml1m.json
│	├── duo4srec
│	│	└── duo4srec_ml1m.json
│	├── ec4srec
│	│	└── ec4srec_ml1m.json
│	└── psac
│		└── psac_gen_ml1m.json
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
│	│		│	├── Digital_Music.data
│	│		│	├── Kindle_Store.data
│	│		│	└── ml.data
│	│		└── lite
│	│			├── All_Beauty_lite.data
│	│			├── Appliances_lite.data
│	│			├── Digital_Music_lite.data
│	│			├── Kindle_Store_lite.data
│	│			└── ml.data
│	├── generator.py
│	├── processor.py
│	├── saved
│	│	├── cl4srec
│	│	│	└── ml1m
│	│	│		└── pre_64_2048_2_2
│	│	│			├── loss.png
│	│	│			├── model.model
│	│	│			└── test_result.json
│	│	├── duo4srec
│	│	│	└── ml1m
│	│	│		└── pre_64_2048_2_2
│	│	└── ec4srec
│	│		└── ml1m
│	│			├── post_64_2048_2_2
│	│			└── pre_64_2048_2_2
│	└── __init__.py
|
├── executor.py
├── lib
│	├── callback.py
│	├── glb_var.py
│	├── graph_util.py
│	├── json_util.py
│	└── util.py
|
├── model
│	├── attnet.py
│	├── framework
│	│	├── cl4srec.py
│	│	├── duo4srec.py
│	│	├── ec4srec.py
│	│	└── psac.py
│	├── loss.py
│	├── __init__.py
│	└── __pycache__
│		├── attnet.cpython-310.pyc
│		├── loss.cpython-310.pyc
│		└── __init__.cpython-310.pyc
├── README.md
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

**Algorithms that have been implemented and algorithms that will be supported soon:**

----------------------------------------------------

CL-based: 

- [x] CL4SRec[[1]](#ref1)	
- [x] Duo4SRec(DuoSRec)[[2]](#ref2)
- [x] EC4SRec[[3]](#ref3)

DL-based:

- [x] PSAC_gen[[4]](#ref4)





## Configure file format

### General

```json
{
	"net":{},
	"seed":{},
    "linux_fast_num_workers":{},
    "email_reminder":{},
	"augmentation":{},
	"dataset":{},
	"train":{},
	"test":{}
}
```

seed: random seed

linux_fast_num_workers: num_workers for dataloader, work when system is Linux 



**Except for the network and augmentation configuration, other configurations are similar, benchmark uses the same other configuration for the same dataset**.

@benchmark configuration for dataset ML-1M on other configuration

```json
{
	...
    "seed": 5566,
    "linux_fast_num_workers": 4,
    "email_reminder":true,
    "dataset": {
        "type": "ml1m",
        "path": "./data/datasets/process/complete/ml.data",
        "crop_or_fill": true,
        "fill_mask": 0,
        "limit_length": 100
    },
    "train": {
        "batch_size": 256,
        "max_epoch": 200,
        "valid_step": 10,
        "gpu_is_available": true,
        "use_amp": true,
        "optimizer_type": "adam",
        "learning_rate": 5e-05,
        "weight_decay": 0.001,
        "betas": [
            0.9,
            0.999
        ],
        "use_lr_schedule": true,
        "lr_max": 0.0001,
        "metric_less": true,
        "save": true,
        "model_save_path": "./data/saved/cl4srec/ml1m/pre_64_2048_2_2/model.model"
    },
    "test": {
        "batch_size": 256,
        "cache_satisfaction_ratio": 0.2,
        "bs_storagy": 1000,
        "slide_T": 4,
        "alter_topk": 10,
        "metrics_at_k": [
            5,
            10,
            15
        ],
        "cache_size": [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ],
        "gpu_is_available": true,
        "save": true
    }
}
```

### CL4SRec

Network and augmentation configuration for CL4SRec on dataset ML-1M

@.\config\cl4srec\cl4srec_ml1m.json

```json
{
    "net": {
        "type": "CL4SRec",
        "is_norm_fist": true,
        "is_net_manual_init": false,
        "is_cl_method": true,
        "d": 64,
        "d_fc": 2048,
        "n_heads": 2,
        "n_layers": 2,
        "posenc_buffer_size": 200,
        "input_types": 3608,
        "mask_item": [
            0
        ],
        "lambda_cl": 0.1,
        "lambda_cls": 0.1
    },
    "augmentation": {
        "type": "Basic",
        "operator": [
            "crop",
            "mask",
            "reorder"
        ],
        "scale": [
            0.4,
            0.4,
            0.4
        ],
        "opr_sample_num": 2,
        "mask_to": 0
    },
    ...
}
```

### Duo4SRec

@.\config\duo4srec\duo4srec_ml1m.json

```json
{
    "net": {
        "type": "Duo4SRec",
        "is_norm_fist": true,
        "is_net_manual_init": false,
        "d": 64,
        "d_fc": 512,
        "n_heads": 2,
        "n_layers": 1,
        "posenc_buffer_size": 200,
        "input_types": 3548,
        "mask_item": [
            0
        ],
        "tau": 0.6,
        "lambda_sl": 0.1,
        "lambda_cls": 0.1
    },
    "augmentation": {
        "type": "Basic",
        "operator": [
            "retrieval"
        ],
        "scale": [
            null
        ],
        "opr_sample_num": 1,
        "mask_to": 0
    },
    ...
}
```

### EC4SRec

@.\config\ec4srec\ec4srec_ml1m.json

```json
{
    "net": {
        "type": "EC4SRec",
        "is_norm_fist": true,
        "is_net_manual_init": false,
        "d": 64,
        "d_fc": 2048,
        "n_heads": 2,
        "n_layers": 2,
        "posenc_buffer_size": 200,
        "input_types": 3549,
        "mask_item": [
            0
        ],
        "impt_score_step": 1,
        "tau": 0.6,
        "lambda_cl": 0.1,
        "lambda_sl_pos": 0.1,
        "lambda_sl_neg": 0.1,
        "lambda_cls": 0.1
    },
    "seed": 5566,
    "augmentation": {
        "type": "EGA",
        "operator": [
            "crop",
            "mask",
            "reorder",
            "retrieval"
        ],
        "scale": [
            0.5,
            0.5,
            0.5,
            null
        ],
        "opr_sample_num": 3,
        "mask_to": 0
    },
    ...
}
```



## Command

### usage

```shell
usage: executor.py [-h] [--data_process DATA_PROCESS] [--config CONFIG] [--saved_model SAVED_MODEL] [--mode MODE]
```

### options

```shell
-h, --help            show this help message and exit
--data_process DATA_PROCESS, -dp DATA_PROCESS
                    type for data process(None/lite/complete)
--config CONFIG, -cfg CONFIG
                    config for run
--saved_model SAVED_MODEL, -sm SAVED_MODEL
                    path for saved model to test
--mode MODE           train/test/train_and_test
```



### quick start

process dataset

```
python executor.py --data_process=lite
python executor.py --data_process=complete
```

CL4SRec on dataset(ml-1m)

```shell
python executor.py -cfg='./config/cl4srec/cl4srec_ml1m.json' --mode=train
python executor.py -cfg='./config/cl4srec/cl4srec_ml1m.json' --mode=train_and_test
python executor.py -sm='./data/saved/cl4srec/ml1m/pre_64_2048_2_2/model.model' --mode=test
```

Duo4SRec on dataset(ml-1m)

```shell
python executor.py -cfg='./config/duo4srec/duo4srec_ml1m.json' --mode=train_and_test
```

EC4SRec on dataset(ml-1m)

```
python executor.py -cfg='./config/ec4srec/ec4srec_ml1m.json' --mode=train_and_test
```



# Refrence

1. <spin id='ref1'></spin>Xie X, Sun F, Liu Z, et al. Contrastive learning for sequential recommendation[C]//2022 IEEE 38th international conference on data engineering (ICDE). IEEE, 2022: 1259-1273.
1. <spin id='ref2'></spin>Qiu R, Huang Z, Yin H, et al. Contrastive learning for representation degeneration problem in sequential recommendation[C]//Proceedings of the fifteenth ACM international conference on web search and data mining. 2022: 813-823.
1. <spin id='ref3'></spin>Wang L, Lim E P, Liu Z, et al. Explanation guided contrastive learning for sequential recommendation[C]//Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022: 2017-2027.
1. <spin id='ref4'></spin>Zhang Y, Li Y, Wang R, et al. PSAC: Proactive sequence-aware content caching via deep learning at the network edge[J]. IEEE Transactions on Network Science and Engineering, 2020, 7(4): 2145-2154.

