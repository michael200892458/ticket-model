#!/bin/sh

cfg="trainer_config_ticket_type.py"

paddle train \
    --config=$cfg \
    --save_dir=./model/ticketTypeModel \
    --trainer_count=4 \
    --log_period=20 \
    --num_passes=15 \
    --use_gpu=false \
    --test_all_data_in_one_period=1 \
    --show_parameter_stats_period=100 \
    2>&1 | tee 'train2.log'
