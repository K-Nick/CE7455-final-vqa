# ce7455-final-vqa

This is the final project for CE7455 Deep Learning in Natural Language Processing

Team Members: Wu Size, Zhang Zhaoqi, Wang Jing

## Project Stucture

Our project is implemented using pytorch lightning and hydra

* configs: including all config files in *.yaml format
* data: code about data preparation and dataloader
* tools: scripts for data preprocessing etc.
* util: common block for file IO and network operation
* model/
  * peterson_base.py // Peterson et al. (simplified)
  * qvcomp // Hadamard Replacement
  * qvguide // Word Level Transformer
  * qvcross // deprecated
  * qvjoint // deprecated, not reported in the final result
* main: the main entry of training
* predict: load checkpoint and collect results from multiple runs

## Usage Instruction

### Environment Requirement

```shell
torch == 1.11.0
pytorch-lightning == 1.6.0
hydra-core == 1.1.1
```

### Train Model

> All code here should run on the GPU server

```shell
python main.py model=... [config parameter]
```

By using hydra, you can adjust any parameter in train.yaml

An example here

```shell
python main.py model=qvguide model.qv_cross.nlayer=3 logger.notes=pe+layernorm
```

### Load Checkpoint & Result Collection

```shell
python predict.py
```

You can adjust the loaded checkpoint under here

```python
ckpt_list = [
    (
        "qvguide-best",
        "./logs/experiments/runs/default/2022-04-19_20-04-38/epoch=50-step=22134.ckpt",
    ),
    (
        "peterson-baseline",
        "./logs/experiments/runs/default/2022-04-16_13-47-18/epoch=35-step=15624.ckpt",
    ),
]
```
