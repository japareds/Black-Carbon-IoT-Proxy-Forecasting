#!/bin/bash
color='\033[1;37m'
color2='\033[1;33m'
nc='\033[0m'
echo -e "${color}--------------------------------------------"
echo "*** BC concentration proxy forecasting ***"
echo -e "--------------------------------------------${nc}"

# Script parameters
device="Ref_st"
source="data_frame"
proxy_fname="experiment_dataSet.pkl"

predictors="BC NO2 O3 N PM25 T RH"
months_train=12
months_val=6
forecasting_hours=0

kernel_size=2 #TDNN hyperparameter

# for low-impact usage: accel='cpu',n_devices=1,num_workers=0

accel="cpu"
n_devices=2 #num of processes (cpu) or num of gpus
num_workers=4 #num of cpus for data loader


batch_size=128
lr=0.0001 #learning rate
wd=0.01 #L2 regularization
scaler="Standard"
Maxepochs=1000 # max epochs

architecture='LSTM'
seqL=(1 2 6 12 24) #sequence length for RNNs
hl=1 # number of hidden layers
dop=0.0 # dropOut probability
nhls=(6 50 100) # nodes per hidden layer
n_output=1 #only 1: BC

echo -e "${color2}Training architecture ${architecture} ${nc}"
for sl in ${seqL[@]}
do
	for nhl in ${nhls[@]}
	do
		echo -e "${color2} sequence length: ${sl}, number of nodes: ${nhl} ${nc}"
		python BCTP.py --train -p ${predictors} -sl ${sl} -me ${Maxepochs} -fh ${forecasting_hours} -lr ${lr} -b ${batch_size} -nd ${n_devices} -nw ${num_workers} -a ${architecture} -hl ${hl} -nhl ${nhl} -dop ${dop} -wd ${wd}
	done
done

