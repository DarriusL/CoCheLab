# Configuration file in CacheLab

There are three configuration files in cachelab:

- data process config

- lab config

- model config

  

## data process config

Configuration files for data processing.

```json
{
    "dp_1":{
        "data_type":"ml1m",
        "src":"./data/datasets/meta/ml-1m/ratings.dat",
        "tgt":"./data/datasets/process/complete/ml.data",
        "crop_or_fill": true,
        "fill_mask": 0,
        "limit_length": 55,
        "repcr_devide_to_train_valid_test":true,
        "ratio":[
            0.6,
            0.2,
            0.2
        ],
        "repcr_tgt":"./data/datasets/process/complete/ml_devide_55.data"
    },
    dp_2:{}
}
```

- dp_1, dp_2: The label of this data processing is only for identification.

- data_type: Provide two data source processing interfaces(movielens/amazon review).

  tips:Because the ml1m format is not consistent with others, please replace the symbol ':' with a space ' ' in advance.

- src:Raw data file path.

- tgt:Original processing data file save path.

- crop_or_fill:Whether to limit the data length.

- fill_mask:If the data length is limited, when the length is less than the limited length, use mask to fill.

- limit_length

- repcr_devide_to_train_valid_test:After the secondary processing (limited length), is it necessary to divide the training set, validation set, and test set?

  tips:Division is generally used for PSAC, Caser

- ratio:Division ratio: [training set, verification set, test set]



## lab config

```json
{
    "email_reminder":{
        "sender":"senderemail@xxx.xxx",
        "password":"Your password or authorization coder",
        "sever":"smtp.xxx.xxx",
        "port":25,
        "receiver":"receivermail@xxx.xxx"
    },
    "constant":{
        "use_amp_true":{
            "eps":1e-7,
            "mask_to_value":-1e+4
        },
        "use_amp_false":{
            "eps":1e-10,
            "mask_to_value":-1e+10
        }
    }
}
```

- email_reminder: Cachelab's mail reminder service configuration.

  To use this service, you need to provide an email address as the sender, and you need to be connected to the Internet.

  - sender: email address of sender.
  - password: password or authorization code of sender.
  - sever: server of sender.
  - port
  - receiver: receiving address

- constant: Configuration of some constants used

  - use_amp_true:When using automatic mixed precision
    - eps
    - mask_to_value
  - use_amp_false:When not using automatic mixed precision
    - eps
    - mask_to_value



## model config

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

- seed: random seed.

- linux_fast_num_workers: num_workers for dataloader, work when system is Linux .
- email_reminder:Whether to enable the email reminder service.



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
        "use_amp": false,
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

- dataset:Dataset related configuration.

  - type:The identifier of the dataset, used to create the path.
  - path:dataset path.
  - crop_or_fill:Whether to limit the data length.
  - fill_mask:If the data length is limited, when the length is less than the limited length, use mask to fill.
  - limit_length

- train:Training related configuration.

  - batch_size

  - max_epoch

  - valid_step: verification interval.

  - gpu_is_available: Whether to use GPU.

  - use_amp: Whether to use automatic mixed precision.

  - optimizer_type: 'adam','adamw'.

    tips: if you need to use other optimizers and add them yourself in the file:@./Room/officer.py/AbstractTrainer.

  - learning_rate: Optimizer initial learning rate.

  - weight_decay

  - betas

  - use_lr_schedule: Whether to use learning rate scheduling

    tips: cachelab uses OneCycleLR, please change it yourself if you use other.

  - lr_max: Learning Rate The maximum learning rate for the scheduler

  - metric_less: Optimization direction, decrease or increase

  - save

  - model_save_path

- test: Testing related configuration.

  - batch_size
  - cache_satisfaction_ratio
  - bs_storagy
  - slide_T
  - alter_topk
  - metrics_at_k
  - gpu_is_available
  - save

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

### PSAC

@.\config\psac\psac_gen_ml1m.json

```json
{
    "net": {
        "type": "PSAC_gen",
        "is_cl_method": false,
        "is_embed_net_manual_init": false,
        "d": 64,
        "n_kernels": 8,
        "L": 3,
        "input_types": 3558,
        "neg_samples": 3,
        "mask_item": [
            0
        ]
    },
    ...
}
```

### Caser

@.\config\caser\caser_ml1m.json

```json
{
    "net": {
        "type": "Caser",
        "is_embed_net_manual_init": false,
        "is_cl_method": false,
        "d": 64,
        "n_kernels": 8,
        "L": 3,
        "input_types": 3558,
        "neg_samples": 3,
        "mask_item": [
            0
        ]
    },
    ...
}
```

