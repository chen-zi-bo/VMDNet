#!/bin/bash

declare -A dataset_to_window_map

#传入参数定义
#模型
models=($1)
#归一化
norms=($2)
#数据集
datasets=($3)
#预测长度的列表
pred_lens=($4)
#设备
device=$5
#输入窗口大小
windows=$6
#归一化配置
norm_config=$7
#四层循环 就是为了根据提供的参数把所有的组合情况都进行训练测试比较
#例如 ./scripts/run_fan_wandb.sh "FEDformer" "VMD_Change"  "Electricity Traffic " "96 168 336 720"  "cuda:0" 96  "{norm_sep_loss:True}"
#                                  模型        归一化    数据集                预测长度列表        设备      输入窗口大小    归一化配置
#                                    该例子循环实际上就有四次
for model in "${models[@]}"
    do
    for norm in "${norms[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for pred_len in "${pred_lens[@]}"
            do
                CUDA_DEVICE_ORDER=PCI_BUS_ID python ./torch_timeseries/norm_experiments/$model.py --dataset_type="$dataset" --norm_type="$norm" --norm_config=$norm_config --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=50 config_wandb --project="VMD_Dlinear_visual" --name="Norm"  runs --seeds='[1]'
#                                                            运行对应的模型的py文件    配置数据集、归一化方式、归一化配置信息、GPU设备，设置批量大小32、预测时间跨度为1、预测时间长度、输入窗口以及训练次数，后面两个是两个调用方法一个config_wandb，一个runs
            done
        done
    done
done

echo "All runs completed."
