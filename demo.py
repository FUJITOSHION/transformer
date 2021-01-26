
import torch.nn as nn
# import numpy as np
# import random

from src.model import TransformerClassification
from src.train_model import train_model
import ipython_ready as ir

dl_class = ir.GetDataLoader()
dl_dict, TEXT = dl_class()
net = TransformerClassification(TEXT.vocab.vectors)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner層の初期化
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# 訓練モードに設定
net.train()

# TransformerBlockモジュールを初期化実行
net.net3_1.apply(weights_init)
net.net3_2.apply(weights_init)

num_epochs = 10
net_trained = train_model(net, dl_dict, num_epoch=num_epochs)
