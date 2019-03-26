#!/bin/bash

export PYTHONPATH="$(pwd)"

python3 main.py --dataset 'cifar10' --is_training True --loss "hinge" --save_dir "./ex0_cifar10_hinge" --gpu 0

python3 main.py --dataset 'cifar10' --is_training False --loss "hinge" --save_dir "./ex0_cifar10_hinge" --gpu 0
