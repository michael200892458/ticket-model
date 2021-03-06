#!/usr/bin/python
# -*- coding:utf-8 -*-

from paddle.trainer_config_helpers import *

CLASS_NUM=7
dict_file = "./data/dict/ticketTypeDict/dict.txt"
word_dict = dict()
with open(dict_file, 'r') as f:
    for i, line in enumerate(f):
        w = line.strip().split()[0]
        word_dict[w] = i

is_predict = get_config_arg('is_predict', bool, False)
trn = 'data/ticketTypeModelData/train.list' if not is_predict else None
tst = 'data/ticketTypeModelData/test.list'  if not is_predict else 'data/ticketTypeModelData/pred.list'
process = 'process' if not is_predict else 'process_predict'
define_py_data_sources2(train_list=trn,
        test_list=tst,
        module="dataprovider_ticket_type",
        obj=process,
        args={"dictionary": word_dict})

batch_size = 128
settings(
        batch_size=batch_size,
        learning_rate=2e-3,
        learning_method=AdamOptimizer(),
        regularization=L2Regularization(8e-4),
        gradient_clipping_threshold=25
        )

data = data_layer(name="word", size=len(word_dict) + 1)
embedding = embedding_layer(input=data, size=128)
conv = sequence_conv_pool(input=embedding, context_len=5, hidden_size=128)
hidden1 = fc_layer(input=conv, size=128)
output = fc_layer(input=hidden1, size=CLASS_NUM, act=SoftmaxActivation())

if not is_predict:
    label = data_layer(name="label", size=CLASS_NUM)
    cls = classification_cost(input=output, label=label, cost='multi_binary_label_cross_entropy')
    outputs(cls)
else:
    outputs([output])

