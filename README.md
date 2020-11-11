

 # Fundamental diagram and volume delay function calibration (traffic_FD_VDF_calibration)

 This program perform a standard procedure to conduct Fundamental diagram (FT) and volume delay function (VDF) calibration. This code focuses on the calibration of the Bureau of Public Roads (BPR) functions The code consists of 5 main steps.

 1. **Input data**
 2. **FT calibration:** For each facility type and area type (a VDF type), we estimate basic coefficients for traffic stream model including free-flow speed, ultimate capacity, critical speed (or cut off speed), and key coefficients m (to control the smoothness or flatness of the curves). Three fundamental diagrams, i.e., speed-density, speed-volume, and volume-density,  are obtained after the calibration 
  3. **Period-based VDF calibration:** For each VDF type and periods, calibrate alpha and beta coefficients in the BPR function
  4. **Daily VDF calibration:** For each VDF type, calibrate daily alpha and beta coefficients in the BPR function
  5. **Output results**



## Input data 

The code read a link performance.csv as its input file. The field names include: 

 ![image-20201111092637996](E:\GitHub\traffic_FD_VDF_calibration\image-20201111092637996.png)

## Important setting 



- REASON_JAM=220 # The upper bound of density (reasonable density jam )

- SPD_CAP

  1: Use speed based hour-to-period factor

   0: use volume based hour-to-period factor

  

- METHOD

  We provide four types of method to calibrate the BPR functions 

  A. “VBM” : volume based method 

  B. “DBM”: density based method 

  C.”QBM”: queue based method 

  D.”BPR_X”: based on the queue based method, use average discharge rate instead of capacity. 

- OUTPUT

  A. "DAY" daily calibrated outputs

  B. “PERIOD” period based calibrated outputs.  

- INCOMP_SAMPLE=0.5 

  if the incomplete of the records larger than the threshold, we will give up the link's data during a time-period

- FILE

  write log file or not. 

- Weight 

  weight_hourly_data: the weight of data of hourly VOC using in calibration  

  weight_period_data: the weight of data of period VOC using in calibration 

  weight_max_cong_period=100: the weight (penalty) of the maximum VOC value 

