#!/usr/bin/python
#coding: utf-8
#
# Copyright (c) 2016 Baidu.com, Inc. All Rights reserved  
#
"""
样本特征抽取器
Authors: liubin04(liubin04@baidu.com)
Date: 16/10/17
"""
from paddle.trainer.PyDataProvider2 import *

UNK_IDX = 0
def initializer(settings, dictionary, **kwargs):
    """
        初始化词典和输入类型
    """
    settings.word_dict = dictionary
    settings.input_types = [
            integer_value_sequence(len(dictionary)),
            integer_value(8)]


@provider(init_hook=initializer, cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, file_name):
    """
        训练集特征抽取
    """
    with open(file_name, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 4:
                continue
            ticketId = fields[0]
            label_str = fields[3]
            if label_str == 'null':
                continue
            labels = []
            try:
                tokens = labelStr.split(',')
                for token in tokens:
                    value = int(token)
                    labels.append(value)
            except:
                continue
            words = fields[2]
            words = words.split(" ");
            word_slot = [settings.word_dict.get(w, UNK_IDX) for w in words]
            yield word_slot, labels


def predict_initializer(settings, dictionary, **kwargs):
    """
        预测的初始化
    """
    settings.word_dict = dictionary
    settings.input_types = [
            integer_value(len(dictionary), seq_type=SequenceType.SEQUENCE)
            ]


@provider(init_hook=predict_initializer, should_shuffle=False)
def process_predict(settings, file_name):
    """
        预测样本特征抽取
    """
    with open(file_name, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 4:
                continue
            ticketId = fields[0]
            words = fields[2]
            words = words.split(" ");
            word_slot = [settings.word_dict.get(w, UNK_IDX) for w in words]
            yield word_slot

