#!/bin/bash
color='\033[1;32m'
nc='\033[0m'
echo -e "${color}---------------------------------------------------"
echo "Recurrent Neural Networks Testing set predictions"
echo -e "---------------------------------------------------${nc}"
# Sctipt parameters
device="Ref_st"
source="data_frame"
proxy_fname="experiment_dataSet.pkl"

predictors="BC NO2 O3 N PM25 T RH"
months_train=12
months_val=6

scaler="Standard"
accel="cpu"
n_devices=2
num_workers=4
bs=128
lr=0.0001

# --------------------------------------------------
# Results of validation for trained networks with:
## months_train=12
## months_val=6
## predictors="BC NO2 O3 N PM25 T RH"
## scaler="Standard"
## seqL = (1 2 6 12 24)hours

#architecture='LSTM'
#seqL=(1 2 6 12 24)
#hls=(1 1 3 1 1)
#nhls=(12 18 12 6 12)
#full_score=(0.68 0.73 0.65 0.65 0.71)
#OF=(N N N N N)
#epochs=(56 20 80 112 21)
#val_scores=(0.67 0.72 0.64 0.65 0.66)

#architecture='GRU'
#seqL=(1 2 6 12 24)
#hls=(1 1 1 1 1)
#nhls=(12 12 12 12 18)
#epochs=(33 30 32 28 57)
#val_scores=(0.64 0.69 0.72 0.68 0.58)
#full_score=(0.65 0.70 0.72 0.69 0.59)
#OF=(N N Y N N)

#architecture='RNN'
#seqL=(1 2 6 12 24)
#hls=(1 1 2 2 1)
#nhls=(12 6 6 6 12)
#epochs=(33 70 53 56 16)
#val_scores=(0.65 0.59 0.65 0.63 0.63)
#full_score=(0.66 0.59 0.66 0.63 0.64)
#OF=(N N N N N)
# ----------------------------------------------

architecture='LSTM'
sl=24
hl=1
nhl=6
earlyStoppingEpoch=40
valLoss=0.62
trainLoss=0.79
forecastHours=0

python BCTP_Parallel.py -p ${predictors} -a ${architecture} -nd ${n_devices} -nw ${num_workers} -bs $bs -lr $lr -sl $sl -hl $hl -nhl $nhl -ese $earlyStoppingEpoch -vl ${valLoss} -tl ${trainLoss} -fh ${forecastHours}





