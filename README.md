# CacheLab

Code for the edge cache algorithm.

## Content

[TOC]



**Algorithms that have been implemented and algorithms that will be supported soon:**

----------------------------------------------------

CL-based: 

- [x] CL4SRec[[1]](#ref1)
- [x] Duo4SRec(DuoSRec)[[2]](#ref2)
- [x] EC4SRec[[3]](#ref3)

DL-based:

- [x] PSAC_gen







## Configure file format

### General

```json
{
	"net":{},
	"seed":{},
	"augmentation":{},
	"dataset":{},
	"train":{},
	"test":{}
}
```

Except for the network and augmentation configuration, other configurations are similar, benchmark uses the same other configuration for the same dataset

@benchmark configuration for dataset ML-1M on other configuration

```json
{
	...
    //Random seed for runtime setting
    "seed": 5566,
    "dataset": {
        //dataset type, easy to identify, does not work in the program
        "type": "ml1m",
        //path to the dataset
        "path": "./data/datasets/process/complete/ml.data",
        //Whether to limit the length of each sequence in the dataset
        "crop_or_fill": true,
        //When the length of the sequence is limited, the mask filled when the length does not reach the limited length,works when crop_or_fill is true
        "fill_mask": 0,
        //The length of the sequence limit, works when crop_or_fill is true
        "limit_length": 100
    },
    "train": {
        //The batch size used by an epoch
        "batch_size": 256,
        //The total number of epochs for training
        "max_epoch": 100,
        //Validation is performed every valid_step
        "valid_step": 10,
        //Whether to use gpu during training
        "gpu_is_available": true,
        //Optimizer for model parameters during training
        //Adam and AdamW is currently supported
        "optimizer_type": "adam",
        //initial learning rate
        "learning_rate": 0.001,
        //weight_decay for regularization
        "weight_decay": 0.001,
        "betas": [
            0.9,
            0.999
        ],
        //whether to use lr schedule, OneCycleLR for here
        "use_lr_schedule": true,
        "lr_max": 0.001,
        //Training trend of the loss function
        "metric_less": true,
        "save": true,
        "model_save_path": "./data/saved/cl4srec/ml1m/pre_64_512_2_1/model.model"
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
        "d": 64,
        "d_q": 32,
        "d_k": 32,
        "d_v": 32,
        "d_fc": 32,
        "n_heads": 2,
        "n_layers": 2,
        "posenc_buffer_size": 200,
        "input_types": 3549,
        "mask_item": [
            0
        ],
        "lambda_cl": 0.2,
        "lambda_cls": 0.5
    },
    "augmentation": {
        "type": "Basic",
        "operator": [
            "crop",
            "mask",
            "reorder"
        ],
        "scale": [
            0.5,
            0.5,
            0.5
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
```



train and test CL4SRec on dataset(ml-1m)

```shell
python executor.py -cfg='./config/cl4srec/cl4srec_ml1m.json' --mode=train_and_test
```

test

```shell
python executor.py -sm='./data/saved/cl4srec/ml1m/pre_64_2048_2_1/model.model' --mode=test
```

train and test Duo4SRec on dataset(ml-1m)

```shell
python executor.py -cfg='./config/duo4srec/duo4srec_ml1m.json' --mode=train_and_test
```



train and test EC4SRec on dataset(ml-1m)

```
python executor.py -cfg='./config/ec4srec/ec4srec_ml1m.json' --mode=train_and_test
```



# Refrence

1. <spin id='ref1'></spin>Xie X, Sun F, Liu Z, et al. Contrastive learning for sequential recommendation[C]//2022 IEEE 38th international conference on data engineering (ICDE). IEEE, 2022: 1259-1273.
1. <spin id='ref2'></spin>Qiu R, Huang Z, Yin H, et al. Contrastive learning for representation degeneration problem in sequential recommendation[C]//Proceedings of the fifteenth ACM international conference on web search and data mining. 2022: 813-823.
1. <spin id='ref3'></spin>Wang L, Lim E P, Liu Z, et al. Explanation guided contrastive learning for sequential recommendation[C]//Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022: 2017-2027.
1. <spin id='ref4'></spin>Zhang Y, Li Y, Wang R, et al. PSAC: Proactive sequence-aware content caching via deep learning at the network edge[J]. IEEE Transactions on Network Science and Engineering, 2020, 7(4): 2145-2154.


