# Black Carbon Temporal Proxy

This code aims to estimate BC concentration (ug/m3) using measurements of other pollutants obtained via IoT devices located 
at Palau Reial Reference Station, Barcelona, Spain.


**Measured quantities: BC, O3, NO2, UFP(N), PM1, PM2.5, PM10, SO2, NOx, CO2, T, RH, Vmax, P**

**Methods implemented: LSTM, GRU, RNN, TDNN (1DCNN)**

**Simple ML algorithms implemented: SVR, RF, MLP, AdaBoost**

Specify parameters on run_train.sh
Visualize validation logger using visualize.sh
Run model on testing set using runt_test.sh
Run MLproxy.sh for non temporal models

Scripts are located in /Scripts folder
Files are located in /Files folder (restricted access)
Results are located in /Results folder (restricted access)

