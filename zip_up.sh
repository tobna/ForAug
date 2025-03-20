#! /bin/bash

path=$(pwd)
cd $1

echo "zip train/foregrounds"
zip -r0 foregrounds_train.zip train/foregrounds | pv -lep -s 1275557 > /dev/null

echo "zip train/backgrounds"
zip -r0 backgrounds_train.zip train/backgrounds | pv -lep -s 1275556 > /dev/null

echo "zip val/foregrounds"
zip -r0 foregrounds_val.zip val/foregrounds | pv -lep -s 50751 > /dev/null

echo "zip val/backgrounds"
zip -r0 backgrounds_val.zip val/backgrounds | pv -lep -s 50751 > /dev/null

cd $path
