# AADA-设备自适应分解算法

本项目旨在构建一种设备自适应负荷分解算法，以提高非侵入负荷监测（NILM）的迁移能力，并解决新房屋、新设备监测性能差的问题。

![aada.md](aada.png)

## 数据
本项目考虑了ukdale, redd, refit 3个低频数据集，且仅考虑kettle, microwave, dishwasher, washing_machine, fridge 5个设备。数据处理逻辑详见$dataset.py$。各数据集介绍如下：
1. ukdale: https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.zip

ukdale数据包含5个房屋的总线和支线功率数据，采样频率为6s。在NILM研究中一般仅使用house_1, house_2, house_5，其中house_1的数据量最多。每个房屋目录下labels.dat存放设备和支线通道的对应关系，其中channel_1为总线功率。

2. redd: http://redd.csail.mit.edu（此链接应该已失效）

redd数据集包含6个房屋的总线和支线功率数据，采样频率为3s，redd的数据量比ukdale少很多。每个房屋目录下labels.dat存放设备和支线通道的对应关系，其中channel_1和channel_2为总线功率。redd数据集中同一电器可能对应多个channel，使用时一般将所对应的所有channel相加。

3. refit: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned

refit数据集包含20个房屋的总线和支线功率数据，采样频率为8s. 每个房屋对应一个csv文件，MetaData_Tables.xlsx存放设备和支线通道的对应关系。


## 训练
本项目训练逻辑见$train.py$，训练集和验证集随机划分，比例为8:2。该脚本下$houses$和$app\_names$指定用于训练的数据集、房屋和电器，如：
```python
houses = { "ukdale": ["house_1", "house_5"] }
app_names = ["kettle", "microwave", "dishwasher", "washing_machine", "fridge"]
```
训练命令如下，其中method指定算法，包括$aada$, $vae$, $s2s$。

```python
python train.py --method aada
```

## 测试
本项目测试逻辑见$test.py$。该脚本下$houses$和$app\_names$指定用于测试的数据集、房屋和电器，如：
```python
houses = { "ukdale": ["house_1", "house_5"] }
app_names = ["kettle", "microwave", "dishwasher", "washing_machine", "fridge"]
```
测试命令如下，其中method指定算法，包括$aada$, $vae$, $s2s$；ckpt指定checkpoint文件名称，该文件格式为{method}-{set}(house_ids)-{appliances}-epoch={epoch}.

```python
python train.py --method aada --ckpt aada-u15-kmdwf-epoch=87
```

## 性能


avae refit256->ukdale2 
| 指标 | Kettle | Microwave| Dishwasher | Washing_meachine | Fridge |
| :----: | :----: | :----: | :----: | :----: | :----: |
| mae | 8.69 | 4.53 | 25.82 | 4.61 | 17.27 |
| mae_on | 778.59 | 782.85 | 806.88 | 277.04 | 19.92 |
| mre_on | 0.26 | 0.62 | 0.68 | 0.68 | 0.22 |

vae refit256->ukdale2 
| 指标 | Kettle | Microwave| Dishwasher | Washing_meachine | Fridge |
| :----: | :----: | :----: | :----: | :----: | :----: |
| mae | 5.98 | 3.91 | 16.35 | 9.72 | 21.76 |
| mae_on | 526.53 | 655.74 | 507.27 | 252.19 | 31.95 |
| mre_on | 0.17 | 0.52 | 0.53 | 0.73 | 0.35 |

avae ukdale15->ukdale2 
| 指标 | Kettle | Microwave| Dishwasher | Washing_meachine | Fridge |
| :----: | :----: | :----: | :----: | :----: | :----: |
| mae | 7.03 | 8.29 | 11.37 | 7.84 | 27.37 |
| mae_on | 661.77 | 828.77 | 332.01 | 462.37 | 43.99 |
| mre_on | 0.22 | 0.64 | 0.47 | 0.77 | 0.49 |


vae ukdale15->ukdale2 
| 指标 | Kettle | Microwave| Dishwasher | Washing_meachine | Fridge |
| :----: | :----: | :----: | :----: | :----: | :----: |
| mae | 7.41 | 5.65 | 11.17 | 3.83 | 15.45 |
| mae_on | 706.32 | 957.19 | 341.46 | 278.48 | 15.04 |
| mre_on | 0.23 | 0.75 | 0.51 | 0.77 | 0.16 |



## 部署
本项目基于fastapi搭建web服务(http://127.0.0.1:8080/predict/)，对外提供调用模型的接口，具体逻辑见app.py。部署命令如下：
```python
python app.py
```
请求时需要使用post请求，以examples和aggs两个参数分别提交示例和总线功率，返回分解结果。如：
```python
axios.post('http://127.0.0.1:8080/predict/', {
    examples: [...],
    aggs: [...]
}).then(res => {
    ...
})
```


## 技术架构

本项目算法部分基于python和lightning实现，$models$目录下存放个算法对应模型的pytorch实现。$lightning\_module.py$下$NilmNet$负责所有模型的训练、验证、测试。models目录结构如下：

```
|-- aada.py: 本文提出的设备自适应分解算法
|-- avae.py: 基于vae架构的设备自适应分解算法
|-- vae.py: 复现的vae实验
|-- s2s.py: 复现的s2s实验
```



## 其他
- $config.py$: 存放与模型和训练相关的配置信息
- $select\_examples.py$: 用于采集各个电器的示例
- $examples$: 用于存放各个电器的示例


