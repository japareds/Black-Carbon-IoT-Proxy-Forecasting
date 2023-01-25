#!/bin/bash
color='\033[1;32m'
nc='\033[0m'
echo -e "${color} ------------------------------------"
echo "Machine Learning Black Carbon proxy"
echo -e " ------------------------------------${nc}"

# Script parameters
device="Ref_st"
source="data_frame"

predictors="BC NO2 O3 N PM25 T RH"
months_train=12
months_val=6
scaler="Standard"

algorithm='MLP'
### Hyperparameters for 12 months of training, 6 of validation and 6 of testing
## SVR
C=10
g=0.001
e=0.1

## RF
ne=3000
md=5
mss=2
ms=0.33
mf=0.5

## MLP
lr=0.01
hs=(12 12)
reg=0.001
python BCTP_simpleProxy.py -mla ${algorithm} -p ${predictors} -m ${months_train} -v ${months_val} -sc ${scaler}  -lr $lr -hs ${hs[@]} -reg $reg
