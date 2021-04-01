# BERT_example

## 介绍

本项目是使用bert微调训练情感二分类模型，采用kaggle[IMDB Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?select=IMDB+Dataset.csv)作为数据集，
预训练模型使用bert_base_uncased，[模型下载地址](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz)，
[词表下载地址](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt)

**目录结构**

```shell
.
├── README.md
├── input
│   ├── bert_base_uncased  # 预训练模型
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   └── imdb.csv  # 数据集
├── models  # 模型存放路径
└── src
    ├── app.py  # 预测接口
    ├── config.py  # 配置文件
    ├── dataset.py  # 输入数据转dataset
    ├── engine.py  # 训练和评估函数
    ├── model.py  # 模型文件
    └── train.py  # 训练调度文件
```

**训练**

下载训练数据和预训练模型，按照目录结构存放，进入src，运行train.py文件
```shell script
python train.py
```

**预测**

进入src，运行app.py，启动flask预测接口
```shell script
python app.py
```

新开一个terminal，输入
```shell script
curl -X POST -H "Content-Type:application/json" -d'{"sentence":"I love you very much"}' http://127.0.0.1:5000/predict
```

即可预测句子情感。