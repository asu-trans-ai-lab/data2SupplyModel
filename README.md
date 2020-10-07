

 # Fundamental diagram and volume delay function calibration (traffic_FD_VDF_calibration)

 This program perform a standard procedure to conduct Fundamental diagram (FT) and volume delay function (VDF) calibration. This code focuses on the calibration of the Bureau of Public Roads (BPR) functions The code consists of 5 main steps.

 1. **Input data**
 2. **FT calibration:** For each facility type and area type (a VDF type), we estimate basic coefficients for traffic stream model including free-flow speed, ultimate capacity, critical speed (or cut off speed), and key coefficients m (to control the smoothness or flatness of the curves). Three fundamental diagrams, i.e., speed-density, speed-volume, and volume-density,  are obtained after the calibration 
  3. **Period-based VDF calibration:** For each VDF type and periods, calibrate alpha and beta coefficients in the BPR function
  4. **Daily VDF calibration:** For each VDF type, calibrate daily alpha and beta coefficients in the BPR function
  5. **Output results**



## Input data 

The code read a link performance.csv as its input file. The field names include: 

 

## FT calibration 

![image-20201007090118596](E:\GitHub\traffic_FD_VDF_calibration\image-20201007090118596.png)