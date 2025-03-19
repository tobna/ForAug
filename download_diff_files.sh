#! /bin/bash

path="`pwd`"
cd $1

wget2 "https://huggingface.co/datasets/TNauen/ForNet/resolve/main/"{train_0.zip,train_1.zip,train_2.zip,train_3.zip,train_4.zip,train_5.zip,train_6.zip,train_7.zip,train_8.zip,train_9.zip,train_10.zip,train_11.zip,train_12.zip,train_13.zip,train_14.zip,train_15.zip,train_16.zip,train_17.zip,train_18.zip,train_19.zip,val.zip,fg_bg_ratios_train.json,fg_bg_ratios_val.json}

cd $path