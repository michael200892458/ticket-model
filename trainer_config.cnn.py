#!/usr/bin/python
#encoding:utf-8
# edit-mode: -*- python -*-
"""
模型结构配置文件
"""
from paddle.trainer_config_helpers import *

dict_file = "../data/peopleGroupModelData/dict.txt"
word_dict = dict()
with open(dict_file, 'r') as f:
    for i, line in enumerate(f):
        w = line.strip().split()[0]
        word_dict[w] = i

is_predict = get_config_arg('is_predict', bool, False)
trn = 'data/peopleGroupModelData/train.list' if not is_predict else None
tst = 'data/peopleGroupModelData/test.list' if not is_predict else 'data/peopleGroupModelData/pred.list'
process = 'process' if not is_predict else 'process_predict'
define_py_data_sources2(train_list=trn,
        test_list=tst,
        module="dataprovider_people_group",
        obj=process,
        args={"dictionary": word_dict})

batch_size = 128 if not is_predict else 1
settings(
        batch_size=batch_size,
        learning_rate=2e-3,
        learning_method=AdamOptimizer(),
        regularization=L2Regularization(8e-4),
        gradient_clipping_threshold=25
        )

data = data_layer(name="word", size=len(word_dict))
embedding = embedding_layer(input=data, size=128)
conv = sequence_conv_pool(input=embedding, context_len=4, hidden_size=512)
output = fc_layer(input=conv, size=8, act=SoftmaxActivation())

if is_predict:
    maxid = maxid_layer(output)
    outputs([maxid, output])
else:
    label = data_layer(name="label", size=8)
    cls = classification_cost(input=output, label=label)
    outputs(cls)

