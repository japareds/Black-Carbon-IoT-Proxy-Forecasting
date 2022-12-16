#!/bin/bash
echo --------------------------------------------
echo "*** BC concentration proxy forecasting ***"
echo --------------------------------------------

# Script parameters
device="Ref_st"
source="data_frame"
proxy_fname="experiment_dataSet.pkl"

predictors="BC NO2 O3 N PM25 T RH"
months_train=12
months_val=6

time_window=6 #sequence length for RNNs
batch_size=128
lr=1e-4
scaler="Standard"

architectures='LSTM RNN GRU'
#hl=3 #number of stacked hidden layers
#n_hl=18 #number of nodes per hidden layer

epochs=5000

kernel_size=2 #only for TDNN architecture
n_output=1 #only 1: BC
dropOut_prob=0.0

# for low-impact usage: accel='cpu',n_devices=1,num_workers=0
accel="cpu"
n_devices=2 #num of processes (cpu) or num of gpus
num_workers=4 #num of cpus for data loader


for architecture in $architectures
do
	echo "Training architecture ${architecture}"
	for hl in {1..3}
	do
		for n_hl in {6..18..6}
		do
			python BCTP_Parallel.py --train -p ${predictors} -sl ${time_window} -a ${architecture} -hl ${hl} -nhl ${n_hl} -lr ${lr} -b${batch_size} -nd ${n_devices} -nw ${num_workers}
		done
	done
done

