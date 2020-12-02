# In[0] Import necessary packages 
import pandas as pd
import matplotlib.pyplot as plt
import os 
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import heapq
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import random 
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import csv

g_number_of_plink=0
g_plink_id_dict={}
g_plink_nb_seq_dict={}
g_parameter_list=[]
g_vdf_group_list=[]

# In[2] Upper bound and lower bound setting
def max_cong_period(period, vdf_name, ASSIGNMENT_PERIOD, UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH):
    nb_period=len(ASSIGNMENT_PERIOD)
    for i in range(nb_period):
        if period == ASSIGNMENT_PERIOD[i]:
            if vdf_name[0] == 1:
                return UPPER_BOUND_DOC_RATIO[i]
            else:
                return PERIOD_LENGTH[i]        
    if period == "Day":
        if vdf_name[0] == 1:
            return np.median(UPPER_BOUND_DOC_RATIO)
        else:
            return np.max(PERIOD_LENGTH)

# In[3] input data
def input_data():
    data_df=pd.read_csv('./link_performance.csv',encoding='UTF-8')
    data_df =data_df.drop(data_df[(data_df.volume == 0) | (data_df.speed == 0)].index) # delete the records speed=0
    #data_df['volume_pl']=data_df['volume']/data_df['lanes']

    # Filtering 
    data_df=data_df[(data_df['FT']==0)|(data_df['FT']==1)|(data_df['FT']==6)]
    #data_df=data_df[(data_df['AT']==1) | (data_df['AT']==2)| (data_df['AT']==3)]
    #data_df=data_df[(data_df['assignment_period']=='1400_1800') ]
    data_df.reset_index(drop=True, inplace=True)

    # Calculate hourly vol and density
    data_df['volume_hourly']=data_df['volume_pl']*(60/TIME_INTERVAL_IN_MIN)
    data_df['density']=data_df['volume_hourly']/data_df['speed']

    return data_df

# In[4] Traffic flow models and volume delay function (BPR function)
def dens_spd_func(x,ffs,k_critical,mm):# fundamental diagram model (density-speed function)
    x_over_k=x/k_critical
    dominator=1+np.power(x_over_k,mm)
    order=2/mm
    return ffs/np.power(dominator,order)

def volume_speed_func(x,ffs,alpha,beta,K_CRI,mm): # fundamental diagram  (volume_delay fuction)
    speed=bpr_func(x,ffs,alpha,beta)
    temp_1=np.power(ffs/speed,mm)
    temp_2=np.power(temp_1,0.5)
    return speed*K_CRI*np.power(temp_2-1,1/mm)

def bpr_func(x,ffs,alpha,beta): # BPR volume delay function
    return ffs/(1+alpha*np.power(x,beta))


# In[5] Calibrate traffic flow model 
def calibrate_traffic_flow(training_set,vdf_name):
    training_set_1=training_set.sort_values(by = 'speed')
    training_set_1.reset_index(drop=True, inplace=True)
    
    lower_bound_FFS=training_set_1['speed'].mean() # The lower bound of freeflow speed (mean value of speed)
    upper_bound_FFS=np.maximum(training_set_1['speed_limit'].mean(),lower_bound_FFS+0.1)  # The upper bound of freeflow speed (mean value of speed limit)


    # fitting speed density fundamental diagram 
    plt.plot(training_set_1['density'], training_set_1['speed'], '*', c='k', label='original values',markersize=2)
    X_data=[]
    Y_data=[]
    for k in range(0,len(training_set_1),10):
        Y_data.append(training_set_1.loc[k:k+10,'speed'].mean())
        if vdf_name[0]==6: 
            threshold=training_set_1.loc[k:k+10,'density'].quantile(0.95) # setting threshold for density
            #threshord_vol=training_set_1.loc[k:k+10,'volume_hourly'].quantile(0.99)
        else:
            threshold=training_set_1.loc[k:k+10,'density'].quantile(0.9) # setting threshold for density
            #threshord_vol=training_set_1.loc[k:k+10,'volume_hourly'].quantile(0.99)
        #intern_training_set_1=training_set_1[k:k+10]
        intern_training_set_1=training_set_1.loc[k:k+10]
        X_data.append(intern_training_set_1[(intern_training_set_1['density']>=threshold)]['density'].mean())
        #X_data.append(intern_training_set_1[(intern_training_set_1['volume_hourly']>=threshord_vol)]['density'].mean())
    x = np.array(X_data)
    y = np.array(Y_data)

    popt,pcov = curve_fit(dens_spd_func, x, y,bounds=[[lower_bound_FFS,0,0],[upper_bound_FFS,UPPER_BOUND_JAM_DENSITY,10]])

    xvals=np.sort(x)
    plt.plot(training_set_1['density'], training_set_1['speed'], '*', c='k', label='original values',markersize=1)
    plt.plot(xvals, dens_spd_func(xvals, *popt), '--',c='r',markersize=6)
    plt.title('Traffic flow function fitting,VDF: '+str(vdf_name[0]+vdf_name[1]*100))
    plt.xlabel('density (vpmpl)')
    plt.ylabel('speed (mph)')
    plt.savefig('./1_FD_speed_density_'+str(vdf_name[0]+vdf_name[1]*100)+'.png')    
    plt.close() 
    
    plt.plot(training_set_1['volume_hourly'], training_set_1['speed'], '*', c='k', label='original values',markersize=1)
    plt.plot(xvals*dens_spd_func(xvals, *popt),dens_spd_func(xvals, *popt), '--',c='r',markersize=6)
    plt.title('Traffic flow function fitting,VDF: '+str(vdf_name[0]+vdf_name[1]*100))
    plt.xlabel('volume (vphpl)')
    plt.ylabel('speed (mph)')
    plt.savefig('./1_FD_speed_volume_'+str(vdf_name[0]+vdf_name[1]*100)+'.png')    
    plt.close() 

    plt.plot(training_set_1['density'], training_set_1['volume_hourly'], '*', c='k', label='original values',markersize=1)
    plt.plot(xvals,xvals*dens_spd_func(xvals, *popt), '--',c='r',markersize=6)
    plt.title('Traffic flow function fitting,VDF: '+str(vdf_name[0]+vdf_name[1]*100))
    plt.xlabel('density (vpmpl)')
    plt.ylabel('volume (vphpl)')
    plt.savefig('./1_FD_volume_density_'+str(vdf_name[0]+vdf_name[1]*100)+'.png')    
    plt.close() 

    FFS=popt[0]
    K_CRI=popt[1]
    mm=popt[2]
    CUT_OFF_SPD=FFS/np.power(2,2/mm)
    ULT_CAP=CUT_OFF_SPD*K_CRI
    print ('--CUT_OFF_SPD=',CUT_OFF_SPD)
    print('--ULT_CAP=',ULT_CAP)
    print ('--K_CRI=',K_CRI)
    print('--FFS=',FFS)
    print('--mm=',mm) 
    return CUT_OFF_SPD,ULT_CAP,K_CRI,FFS,mm

# In[8] VDF calibration 
#def vdf_calculation(internal_vdf_dlink_df,vdf_name,period_name,CUT_OFF_SPD,ULT_CAP,K_CRI,FFS,mm,peak_factor_avg):
def vdf_calculation(internal_vdf_dlink_df, vdf_name, period_name, CUT_OFF_SPD, ULT_CAP, K_CRI, FFS, mm,
                    peak_factor_avg, ASSIGNMENT_PERIOD, UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH):   
    internal_vdf_dlink_df['VOC_period']=internal_vdf_dlink_df['vol_period'].mean()/(ULT_CAP*peak_factor_avg)
    p0=np.array([FFS,0.15,4])
    lowerbound_fitting=[FFS,0.15,1.01] # upper bound and lower bound of free flow speed, alpha and beta
    upperbound_fitting=[FFS*1.1,10,10]

    popt_1=np.array([K_CRI,mm])
    if DOC_RATIO_METHOD =='VBM':
        print('Volume method calibration...')
        internal_vdf_dlink_df['VOC']=internal_vdf_dlink_df.apply(lambda x: (ULT_CAP+(ULT_CAP-x.vol_period_hourly))/ULT_CAP if x.speed_period<CUT_OFF_SPD else x.vol_period_hourly/ULT_CAP,axis=1)

        X_data=[]
        Y_data=[]

        for k in range(0,len(internal_vdf_dlink_df)):
            # Hourly VOC data 
            for kk in range(WEIGHT_HOURLY_DATA):
                Y_data.append(internal_vdf_dlink_df.loc[k,'speed_period'])
                X_data.append(internal_vdf_dlink_df.loc[k,'VOC'])
            # Period VOC data
            for kk in range(WEIGHT_PERIOD_DATA):
                Y_data.append(internal_vdf_dlink_df['speed_period'].mean())
                X_data.append(internal_vdf_dlink_df['VOC_period'].mean())
            for kk in range(WEIGHT_UPPER_BOUND_DOC_RATIO):
                Y_data.append(0.001)
                X_data.append(
                    max_cong_period(period_name, vdf_name, ASSIGNMENT_PERIOD, UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH))


        x = np.array(X_data)
        y = np.array(Y_data)
        popt_VBM,pcov = curve_fit(bpr_func, x, y,p0, bounds=[lowerbound_fitting,upperbound_fitting])
        VBM_RMSE=np.power((np.sum(np.power((bpr_func(x, *popt_VBM)-y),2))/len(x)),0.5)
        VBM_RSE=np.sum(np.power((bpr_func(x, *popt_VBM)-y),2))/np.sum(np.power((bpr_func(x, *popt_VBM)-y.mean()),2))

        xvals=np.linspace(0,5,50)

        plt.plot(x, y, '*', c='k', label='original values',markersize=3)

        plt.plot(xvals,bpr_func(xvals, *popt_VBM), '--',c='r',markersize=6)
        plt.plot(volume_speed_func(xvals,*popt_VBM,*popt_1)/ULT_CAP,bpr_func(xvals, *popt_VBM),c='b')
        VBM_PE=np.mean(np.abs((bpr_func(x, *popt_VBM)-y)/y))
        #print('VBM,'+str(vdf_name)+' '+str(period_name)+',RMSE='+str(round(VBM_RMSE,2))+'RSE='+str(round(VBM_RSE,2)))
        plt.title('VBM,'+str(vdf_name[0]+vdf_name[1]*100)+' '+str(period_name)+',RSE='+str(round(VBM_RSE,3))+'% ,ffs='+str(round(popt_VBM[0],2))+',alpha='+str(round(popt_VBM[1],2))+',beta='+str(round(popt_VBM[2],2)))
        plt.xlabel('VOC')
        plt.ylabel('speed (mph)')
        plt.savefig('./2_VDF_VBM_'+str(vdf_name[0]+vdf_name[1]*100)+'_'+str(period_name)+'.png')    
        plt.close() 
        internal_vdf_dlink_df['alpha']=round(popt_VBM[1],2)
        internal_vdf_dlink_df['beta']=round(popt_VBM[2],2)
        
    #-------------------------------------
    if DOC_RATIO_METHOD =='DBM':
        print('Density method calibration...')
        internal_vdf_dlink_df['VOC']=internal_vdf_dlink_df.apply(lambda x: x.density_period/K_CRI,axis=1)
        X_data=[]
        Y_data=[]
        for k in range(0,len(internal_vdf_dlink_df)):
            # Hourly VOC data 
            for kk in range(WEIGHT_HOURLY_DATA):
                Y_data.append(internal_vdf_dlink_df.loc[k,'speed_period'])
                X_data.append(internal_vdf_dlink_df.loc[k,'VOC'])
            # Period VOC data
            for kk in range(WEIGHT_PERIOD_DATA):
                Y_data.append(internal_vdf_dlink_df['speed_period'].mean())
                X_data.append(internal_vdf_dlink_df['VOC_period'].mean())
            for kk in range(WEIGHT_UPPER_BOUND_DOC_RATIO):
                Y_data.append(0.001)
                X_data.append(max_cong_period(period_name, vdf_name, ASSIGNMENT_PERIOD, UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH))


        x = np.array(X_data)
        y = np.array(Y_data)
        popt_DBM,pcov = curve_fit(bpr_func, x, y, p0, bounds=[lowerbound_fitting,upperbound_fitting])
        DBM_RMSE=np.power((np.sum(np.power((bpr_func(x, *popt_DBM)-y),2))/len(x)),0.5)
        DBM_RSE=np.sum(np.power((bpr_func(x, *popt_DBM)-y),2))/np.sum(np.power((bpr_func(x, *popt_DBM)-y.mean()),2))

        xvals=np.linspace(0,5,50)
        plt.plot(x, y, '*', c='k', label='original values',markersize=3)

        plt.plot(xvals,bpr_func(xvals, *popt_DBM), '--',c='r',markersize=6)
        plt.plot(volume_speed_func(xvals,*popt_DBM,*popt_1)/ULT_CAP,bpr_func(xvals, *popt_DBM),c='b')
        DBM_PE=np.mean(np.abs((bpr_func(x, *popt_DBM)-y)/y))
        #print('DBM,'+str(vdf_name)+' '+str(period_name)+',RMSE='+str(round(DBM_RMSE,2))+'RSE='+str(round(DBM_RSE,2)))
        plt.title('DBM,'+str(vdf_name[0]+vdf_name[1]*100)+' '+str(period_name)+',RSE='+str(round(DBM_RSE,2)) +'% ,ffs='+str(round(popt_DBM[0],2))+',alpha='+str(round(popt_DBM[1],2))+',beta='+str(round(popt_DBM[2],2)))
        plt.xlabel('VOC')
        plt.ylabel('speed (mph)')
        plt.savefig('./2_VDF_DBM_'+str(vdf_name[0]+vdf_name[1]*100)+'_'+str(period_name)+'.png')    
        plt.close() 
        internal_vdf_dlink_df['alpha']=round(popt_DBM[1],2)
        internal_vdf_dlink_df['beta']=round(popt_DBM[2],2)   


    if DOC_RATIO_METHOD =='QBM':
        print('Queue based method method calibration...')
        internal_vdf_dlink_df['VOC']=internal_vdf_dlink_df.apply(lambda x: x.Demand/ULT_CAP,axis=1)


        X_data=[]
        Y_data=[]
        for k in range(0,len(internal_vdf_dlink_df)):
            # Hourly VOC data 
            for kk in range(WEIGHT_HOURLY_DATA):
                Y_data.append(internal_vdf_dlink_df.loc[k,'speed_period'])
                X_data.append(internal_vdf_dlink_df.loc[k,'VOC'])
            # Period VOC data
            for kk in range(WEIGHT_PERIOD_DATA):
                Y_data.append(internal_vdf_dlink_df['speed_period'].mean())
                X_data.append(internal_vdf_dlink_df['VOC_period'].mean())
            for kk in range(WEIGHT_UPPER_BOUND_DOC_RATIO):
                Y_data.append(0.001)
                X_data.append(max_cong_period(period_name, vdf_name, ASSIGNMENT_PERIOD, UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH))


        x = np.array(X_data)
        y = np.array(Y_data)
        popt_QBM,pcov = curve_fit(bpr_func, x, y,bounds=[lowerbound_fitting,upperbound_fitting])
        QBM_RMSE=np.power((np.sum(np.power((bpr_func(x, *popt_QBM)-y),2))/len(x)),0.5)
        QBM_RSE=np.sum(np.power((bpr_func(x, *popt_QBM)-y),2))/np.sum(np.power((bpr_func(x, *popt_QBM)-y.mean()),2))

        xvals=np.linspace(0,5,50)
        plt.plot(x, y, '*', c='k', label='original values',markersize=3)
        plt.plot(xvals,bpr_func(xvals, *popt_QBM), '--',c='r',markersize=6)
        plt.plot(volume_speed_func(xvals,*popt_QBM,*popt_1)/ULT_CAP,bpr_func(xvals, *popt_QBM),c='b')
        
        QBM_PE=np.mean(np.abs((bpr_func(x, *popt_QBM)-y)/y))
        #print('QBM,'+str(vdf_name)+' '+str(period_name)+',RMSE='+str(round(QBM_RMSE,2))+'RSE='+str(round(QBM_RSE,2)))
        plt.title('QBM,'+str(vdf_name[0]+vdf_name[1]*100)+' '+str(period_name)+',RSE='+str(round(QBM_RSE,2))+'%,ffs='+str(round(popt_QBM[0],2))+',alpha='+str(round(popt_QBM[1],2))+',beta='+str(round(popt_QBM[2],2)))
        plt.xlabel('VOC')
        plt.ylabel('speed (mph)')
        plt.savefig('./2_VDF_QBM_'+str(vdf_name[0]+vdf_name[1]*100)+'_'+str(period_name)+'.png')    
        plt.close() 
        internal_vdf_dlink_df['alpha']=round(popt_QBM[1],2)
        internal_vdf_dlink_df['beta']=round(popt_QBM[2],2)   


    if DOC_RATIO_METHOD =='BPR_X':
        print('BPR_X method calibration...')
        internal_vdf_dlink_df['VOC']=internal_vdf_dlink_df.apply(lambda x: x.Demand/x.avg_discharge_rate if x.congestion_period>=PSTW else x.Demand/ULT_CAP, axis=1 )

        X_data=[]
        Y_data=[]
        for k in range(0,len(internal_vdf_dlink_df)):
            # Hourly VOC data 
            for kk in range(WEIGHT_HOURLY_DATA):
                Y_data.append(internal_vdf_dlink_df.loc[k,'speed_period'])
                X_data.append(internal_vdf_dlink_df.loc[k,'VOC'])
            # Period VOC data
            for kk in range(WEIGHT_PERIOD_DATA):
                Y_data.append(internal_vdf_dlink_df['speed_period'].mean())
                X_data.append(internal_vdf_dlink_df['VOC_period'].mean())
            for kk in range(WEIGHT_UPPER_BOUND_DOC_RATIO):
                Y_data.append(0.001)
                X_data.append(max_cong_period(period_name, vdf_name, ASSIGNMENT_PERIOD, UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH))

        x = np.array(X_data)
        y = np.array(Y_data)
        popt_BPR_X,pcov = curve_fit(bpr_func, x, y,bounds=[lowerbound_fitting,upperbound_fitting])
        BPR_X_RMSE=np.power((np.sum(np.power((bpr_func(x, *popt_BPR_X)-y),2))/len(x)),0.5)
        BPR_X_RSE=np.sum(np.power((bpr_func(x, *popt_BPR_X)-y),2))/np.sum(np.power((bpr_func(x, *popt_BPR_X)-y.mean()),2))

        xvals=np.linspace(0,5,50)
        plt.plot(x, y, '*', c='k', label='original values',markersize=3)
        plt.plot(xvals,bpr_func(xvals, *popt_BPR_X), '--',c='r',markersize=6)
        plt.plot(volume_speed_func(xvals,*popt_BPR_X,*popt_1)/ULT_CAP,bpr_func(xvals, *popt_BPR_X),c='b')
        
        BPR_X_PE=np.mean(np.abs((bpr_func(x, *popt_BPR_X)-y)/y))
        #print('BPR_X,'+str(vdf_name)+' '+str(period_name)+',RMSE='+str(round(BPR_X_RMSE,2))+'RSE='+str(round(BPR_X_RSE,2)))
        plt.title('BPR_X,'+str(vdf_name[0]+vdf_name[1]*100)+' '+str(period_name)+',RSE='+str(round(BPR_X_RSE,2))+'%,ffs='+str(round(popt_BPR_X[0],2))+',alpha='+str(round(popt_BPR_X[1],2))+',beta='+str(round(popt_BPR_X[2],2)))
        plt.xlabel('VOC')
        plt.ylabel('speed (mph)')
        plt.savefig('./2_VDF_BPR_X '+str(vdf_name[0]+vdf_name[1]*100)+'_'+str(period_name)+'.png')    
        plt.close() 
        internal_vdf_dlink_df['alpha']=round(popt_BPR_X[1],2)
        internal_vdf_dlink_df['beta']=round(popt_BPR_X[2],2)   

    return internal_vdf_dlink_df

def vdf_calculation_daily(temp_daily_df, vdf_name, CUT_OFF_SPD, ULT_CAP, K_CRI, FFS, mm, ASSIGNMENT_PERIOD,
                          UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH):
    p0=np.array([FFS,0.15,4])
    lowerbound_fitting=[FFS,0.15,1.01]
    upperbound_fitting=[FFS*1.1,10,10]
    popt_1=np.array([K_CRI,mm])

    X_data=[]
    Y_data=[]
   
    for k in range(0,len(temp_daily_df)):
        # Hourly VOC data 
        for kk in range(WEIGHT_HOURLY_DATA):
            Y_data.append(temp_daily_df.loc[k,'speed_period'])
            X_data.append(temp_daily_df.loc[k,'VOC'])
        # Period VOC data
            # Period VOC data
        for kk in range(WEIGHT_PERIOD_DATA):
            Y_data.append(temp_daily_df['speed_period'].mean())
            X_data.append(temp_daily_df['VOC_period'].mean())
        for kk in range(WEIGHT_UPPER_BOUND_DOC_RATIO):
            Y_data.append(0.001)
            X_data.append(max_cong_period("Day", vdf_name, ASSIGNMENT_PERIOD, UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH))
        x = np.array(X_data)
        y = np.array(Y_data)

    popt_daily,pcov = curve_fit(bpr_func, x, y,bounds=[lowerbound_fitting,upperbound_fitting])
    daily_RMSE=np.power((np.sum(np.power((bpr_func(x, *popt_daily)-y),2))/len(x)),0.5)
    daily_RSE=np.sum(np.power((bpr_func(x, *popt_daily)-y),2))/np.sum(np.power((bpr_func(x, *popt_daily)-y.mean()),2))

    xvals=np.linspace(0,5,50)
    plt.plot(x, y, '*', c='k', label='original values',markersize=3)
    plt.plot(xvals,bpr_func(xvals, *popt_daily), '--',c='r',markersize=6)
    plt.plot(volume_speed_func(xvals,*popt_daily,*popt_1)/ULT_CAP,bpr_func(xvals, *popt_daily),c='b')
        
    daily_PE=np.mean(np.abs((bpr_func(x, *popt_daily)-y)/y))
    #print('Daily_'+DOC_RATIO_METHOD+','+str(vdf_name)+',RMSE='+str(round(daily_RMSE,2))+'RSE='+str(round(daily_RSE,2)))
    plt.title('Daily_'+DOC_RATIO_METHOD+','+str(vdf_name[0]+vdf_name[1]*100)+',RSE='+str(round(daily_RSE,2))+'%,ffs='+str(round(popt_daily[0],2))+',alpha='+str(round(popt_daily[1],2))+',beta='+str(round(popt_daily[2],2)))
    plt.xlabel('VOC')
    plt.ylabel('speed (mph)')
    plt.savefig('./2_VDF_'+DOC_RATIO_METHOD+'_'+str(vdf_name[0]+vdf_name[1]*100)+'_day.png')    
    plt.close()
    alpha_dict[temp_daily_df.VDF_TYPE.unique()[0]]=round(popt_daily[1],2)
    beta_dict[temp_daily_df.VDF_TYPE.unique()[0]]=round(popt_daily[2],2)
    temp_daily_df['alpha_day']=round(popt_daily[1],2)
    temp_daily_df['beta_day']=round(popt_daily[2],2)   

    return alpha_dict,beta_dict,temp_daily_df


# In[9] Calculate demand and congestion period
def calculate_congestion_period(speed_15min,volume_15min,CUT_OFF_SPD,ULT_CAP):
    global PSTW
    nb_time_stamp=len(speed_15min)
    min_speed=min(speed_15min)
    min_index=speed_15min.index(min(speed_15min)) # The index of speed with minimum value 
    
    # start time and ending time of prefered service time window
    PSTW_st=max(min_index-2,0)
    PSTW_ed=min(min_index+1,nb_time_stamp)
    if PSTW_ed - PSTW_st < 3:
        if PSTW_st==0:
            PSTW_ed=PSTW_ed+(3-(PSTW_ed - PSTW_st))
        if PSTW_ed==nb_time_stamp:
            PSTW_st=PSTW_st-(3-(PSTW_ed - PSTW_st))
    PSTW=(PSTW_ed-PSTW_st+1)*(TIME_INTERVAL_IN_MIN/60)
    PSTW_volume=np.array(volume_15min[PSTW_st:PSTW_ed+1]).sum()
    PSTW_speed=np.array(speed_15min[PSTW_st:PSTW_ed+1]).mean()

    # Determine 
    t3=nb_time_stamp-1
    t0=0
    if min_speed<=CUT_OFF_SPD:
        for i in range(min_index,nb_time_stamp):
            if speed_15min[i]>CUT_OFF_SPD:               
                t3=i-1
                break
        for j in range(min_index,-1,-1):
            #t0=PSTW_st
            if speed_15min[j]>CUT_OFF_SPD:               
                t0=j+1
                break
    elif min_speed >CUT_OFF_SPD:
        t0=0
        t3=0
    congestion_period=(t3-t0+1)*(TIME_INTERVAL_IN_MIN/60)
    Mu=np.mean(volume_hour[t0:t3+1])
    #gamma=(plink.waiting_time.mean()*120*plink.Mu)/np.power(plink.congestion_period,4)   
    
    if congestion_period>PSTW:
        Demand=np.array(volume_15min[t0:t3+1]).sum()
        speed_period=np.array(speed_15min[t0:t3+1]).mean()
    elif congestion_period<=PSTW:
        Demand=PSTW_volume
        speed_period=PSTW_speed

    return t0, t3,congestion_period,PSTW_st,PSTW_ed,PSTW,Demand,Mu,speed_period

# In[10] Validations
def validation(ffs,alpha,beta,K_CRI,mm,volume,capacity):
    u_assign=ffs/(1+alpha*np.power(volume/capacity,beta))
    A=np.power(np.power(ffs/u_assign,mm),0.5)
    flow_assign=u_assign*K_CRI*np.power(A-1,1/mm)
    
    return flow_assign

# In[11] Check whether the samples are complete 
def nb_sample_checking(period, ASSIGNMENT_PERIOD, UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH):
    nb_period=len(ASSIGNMENT_PERIOD)
    for i in range(nb_period):
        if period == ASSIGNMENT_PERIOD[i]:
            return PERIOD_LENGTH[i] * (60 / TIME_INTERVAL_IN_MIN)


# In[12] Main function
if  __name__ == "__main__":
    # In[1] Set parameters. 
    with open('./setting.csv',encoding='UTF-8') as setting_file:
        setting_csv = csv.reader(setting_file)
        for row in setting_csv:
            if row[0] == "DOC_RATIO_METHOD": # volume based method, VBM; density based method, DBM, and queue based method QBM, or BPR_X
                DOC_RATIO_METHOD = row[1]
            if row[0] == "OUTPUT": # DAY, or PERIOD
                OUTPUT = row[1]
            if row [0] =="PHF_METHOD": # method to calculate the peak hour factor 
                PHF_METHOD = row[1] # volume based method, VBM; speed based method SBM
            if row [0] == "LOG_FILE":
                LOG_FILE= int(row[1])
            if row[0] == "TIME_INTERVAL_IN_MIN":
                TIME_INTERVAL_IN_MIN = float(row[1])
            if row[0] == "UPPER_BOUND_JAM_DENSITY": # upper bound of the jam density
                UPPER_BOUND_JAM_DENSITY = float(row[1])
            if row [0] == "MIN_THRESHOLD_SAMPLING":
                MIN_THRESHOLD_SAMPLING = float(row[1])
            if row [0] == "WEIGHT_HOURLY_DATA":
                WEIGHT_HOURLY_DATA = int(row[1])
            if row [0] == "WEIGHT_PERIOD_DATA":
                WEIGHT_PERIOD_DATA = int(row[1])
            if row [0] == "WEIGHT_UPPER_BOUND_DOC_RATIO":
                WEIGHT_UPPER_BOUND_DOC_RATIO= int(row[1])
            if row[0] =="ASSIGNMENT_PERIOD":
                ASSIGNMENT_PERIOD = row[1].split(";")
            if row[0] =="UPPER_BOUND_DOC_RATIO":
                UPPER_BOUND_DOC_RATIO = list(map(float,row[1].split(";"))) # set the upper bound of congestion period 

    
    PERIOD_LENGTH=[]
    for jj in ASSIGNMENT_PERIOD:
        time_ss = [int(var[0:2]) for var in jj.split('_')]
        if time_ss[0] > time_ss[1]:
            Interval_value = time_ss[1] + 24 - time_ss[0]
        else:
            Interval_value = time_ss[1] - time_ss[0]
        PERIOD_LENGTH.append(Interval_value)


    # Step 1: Input data...
    if LOG_FILE ==1: 
        log_file = open("./log.txt", "w")
        log_file.truncate()
        log_file.write('Step 1:Input data...\n')
    
    print('Step 1:Input data...')
    start_time=time.time()

    training_set=input_data()
    # Calculate the length of period and max congest period



    end_time=time.time()
    print('CPU time:',end_time-start_time,'s\n')
    
    if LOG_FILE ==1: 
        log_file.write('CPU time:'+ str(end_time-start_time)+'s\n\n')

    # Group based on VDF types...
    FT_set=training_set['FT'].unique()
    AT_set=training_set['AT'].unique()
    vdf_group=training_set.groupby(['FT','AT']) # Group by VDF types 
    output_df_daily=pd.DataFrame() # build up empty dataframe
    output_df=pd.DataFrame() # build up empty dataframe
    alpha_dict={}
    beta_dict={}

    iter = 0
    for vdf_name,vdf_trainingset in vdf_group: 
        temp_daily_df=pd.DataFrame() # build up empty dataframe
        # Step 2: For each VDF, calibrate basic coefficients for fundamental diagrams
        print('Step 2: Calibrate'+str(vdf_name)+' key coefficients...')
        if LOG_FILE ==1: 
            log_file.write('Step 2: Calibrate'+str(vdf_name)+' key coefficients...\n')

        start_time=time.time()
        vdf_trainingset.reset_index(drop=True, inplace=True)
        CUT_OFF_SPD,ULT_CAP,K_CRI,FFS,mm=calibrate_traffic_flow(vdf_trainingset,vdf_name) # calibrate parameters of traffic flow model

        end_time=time.time()

        print('CPU time:',end_time-start_time,'s\n')
        
        if LOG_FILE ==1: 
            log_file.write('CPU time:'+str(end_time-start_time)+'s\n\n')

        # Step 3: For each VDF and period, calibrate alpha and beta
        print('Step 3: Calibrate VDF function of links for VDF_type: '+str(vdf_name)+' and time period...')
        if LOG_FILE ==1: 
            log_file.write('Step 3: Calibrate VDF function of links for VDF_type: '+str(vdf_name)+' and time period...\n')

        start_time=time.time()
        # 
        pvdf_group=vdf_trainingset.groupby(['assignment_period'])
        for period_name, pvdf_trainset in pvdf_group:
            internal_vdf_dlink_df= pd.DataFrame()
            
            dlink_group=pvdf_trainset.groupby(['link_id','from_node_id','to_node_id','Date'])
            vdf_dlink_list=[]
            
            # Step 3.1 Calculate the VOC (congestion period)
            print('Step 3.1: Calculate the VOCs of links: '+str(vdf_name)+' and time period '+str(period_name))
            if LOG_FILE ==1: 
                log_file.write('Step 3.1: Calculate the VOCs of links: '+str(vdf_name)+' and time period '+str(period_name)+'\n')

            for dlink_name,dlink_training_set in dlink_group:
                dlink_id=dlink_name[0]
                from_node_id=dlink_name[1]
                to_node_id=dlink_name[2]
                date=dlink_name[3]
                FT=vdf_name[0]
                AT=vdf_name[1]
                period=period_name
                vol_period=dlink_training_set['volume_pl'].sum() # summation of all volume within the period
                vol_period_hourly=dlink_training_set['volume_hourly'].mean()
                speed_period=dlink_training_set['speed'].mean()
                density_period=dlink_training_set['density'].mean()
                if len(dlink_training_set) < nb_sample_checking(period_name, ASSIGNMENT_PERIOD, UPPER_BOUND_DOC_RATIO,PERIOD_LENGTH):
                    print('WARNING:  link ', dlink_id, 'in period', period_name,'does not have all 15 minutes records...')
                    print((1 - len(dlink_training_set) / nb_sample_checking(period_name, ASSIGNMENT_PERIOD,UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH)) * 100,
                          '% of records of the link of the time period are missing...\n')
                    if (1 - len(dlink_training_set) / nb_sample_checking(period_name, ASSIGNMENT_PERIOD,UPPER_BOUND_DOC_RATIO,
                                                                         PERIOD_LENGTH)) >= MIN_THRESHOLD_SAMPLING:
                        continue
                
                volume_15min=dlink_training_set.volume_pl.to_list()
                speed_15min=dlink_training_set.speed.to_list()
                volume_hour=dlink_training_set.volume_hourly.to_list()
                
                # Step 4.1 Calculate VOC
                t0, t3,congestion_period,PSTW_st,PSTW_ed,PSTW,Demand,avg_discharge_rate,speed_period_1=calculate_congestion_period(speed_15min,volume_15min,CUT_OFF_SPD,ULT_CAP)
                # d_over_c_bprx is the VOC for queue-based method 
                # Calculate peak hour factor for each link
                vol_hour_max=np.max(volume_hour)
                EPS = ULT_CAP/7 # setting a lower bound of demand 
                if PHF_METHOD=='SBM':
                    peak_hour_factor=vol_period/max(Demand,EPS)
                    if peak_hour_factor ==1:
                        print('WARNING: peak hour factor is 1,delete the link')
                        continue
                if PHF_METHOD=='VBM':
                #vol_hour_max=np.max(volume_hour)
                    peak_hour_factor=vol_period/vol_hour_max      
                
                dlink=[dlink_id,from_node_id, to_node_id,date,FT,AT,period,vol_period, vol_period_hourly,\
                    speed_period,density_period,t0,t3,Demand,avg_discharge_rate,peak_hour_factor,congestion_period]
                vdf_dlink_list.append(dlink)

            
            internal_vdf_dlink_df= pd.DataFrame(vdf_dlink_list)
            internal_vdf_dlink_df.rename(columns={0:'link_id',
                                    1:'from_node_id',
                                    2:'to_node_id',
                                    3:'Date',
                                    4:'FT',
                                    5:'AT',
                                    6:'period',
                                    7:'vol_period',
                                    8:'vol_period_hourly',
                                    9:'speed_period', 
                                    10: 'density_period',                                            
                                    11:'t0',
                                    12:'t3',
                                    13:'Demand',
                                    14:'avg_discharge_rate',
                                    15:'peak_hour_factor',
                                    16:'congestion_period'}, inplace=True)
            internal_vdf_dlink_df.to_csv('./1_'+str(100*vdf_name[1]+vdf_name[0])+','+str(period_name)+'training_set.csv',index=False)
            peak_factor_avg=np.mean(internal_vdf_dlink_df.peak_hour_factor) 
            
            # Step 4.2 VDF calibration
            print('Step 3.2 :VDF calibration: '+str(vdf_name)+' and time period '+str(period_name))
            if LOG_FILE ==1: 
                log_file.write('Step 3.2 :VDF calibration: '+str(vdf_name)+' and time period '+str(period_name)+'\n')

            calibration_vdf_dlink_results = vdf_calculation(internal_vdf_dlink_df, vdf_name, period_name, CUT_OFF_SPD,
                                                            ULT_CAP, K_CRI, FFS, mm, peak_factor_avg, ASSIGNMENT_PERIOD,
                                                            UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH)
            #calibration_vdf_dlink_results.to_csv('./1_'+str(100*vdf_name[1]+vdf_name[0])+'_'+str(period_name)+'training_output.csv',index=False)
            grouyby_results=calibration_vdf_dlink_results.groupby(['link_id','from_node_id','to_node_id','FT','AT','period'])
            vdf_link_list=[]
            for link_name,calibration_outputs in grouyby_results:
                link_id=link_name[0]
                from_node_id=link_name[1]
                to_node_id=link_name[2]
                FT=link_name[3]
                AT=link_name[4]
                period=link_name[5]
                vol_period=calibration_outputs.vol_period.mean()
                vol_period_hourly=calibration_outputs.vol_period_hourly.mean()
                speed_period=calibration_outputs.speed_period.mean()
                density_period=calibration_outputs.density_period.mean()
                t0=calibration_outputs.t0.mean()
                t3=calibration_outputs.t3.mean()
                Demand=calibration_outputs.Demand.mean()
                VOC=calibration_outputs.VOC.mean()
                VOC_period=calibration_outputs.VOC_period.mean()
                alpha=calibration_outputs.alpha.mean()
                beta=calibration_outputs.beta.mean()
                peak_hour_factor=calibration_outputs.peak_hour_factor.mean()
                period_capacity=peak_hour_factor*ULT_CAP
                vol_valid=validation(FFS,alpha,beta,K_CRI,mm,vol_period,period_capacity)
                demand_est=VOC*period_capacity
                vdf_link=[link_id,from_node_id, to_node_id,FT,AT,period,vol_period, vol_period_hourly,speed_period,density_period,t0,t3,Demand,VOC,\
                    VOC_period,alpha,beta,peak_hour_factor,period_capacity,vol_valid,demand_est]
                vdf_link_list.append(vdf_link)
            
            internal_vdf_link_df= pd.DataFrame(vdf_link_list)
            internal_vdf_link_df.rename(columns={0:'link_id',
                                    1:'from_node_id',
                                    2:'to_node_id',
                                    3:'FT',
                                    4:'AT',
                                    5:'period',
                                    6:'vol_period',
                                    7:'vol_period_hourly',
                                    8:'speed_period', 
                                    9: 'density_period',                                            
                                    10:'t0',
                                    11:'t3',
                                    12:'Demand',
                                    13:'VOC',
                                    14:'VOC_period',
                                    15:'alpha',
                                    16:'beta',
                                    17:'peak_hour_factor',
                                    18:'period_cap',                                    
                                    19:'vol_valid',
                                    20:'demand_est'}, inplace=True)
            

            temp_daily_df = pd.concat([temp_daily_df,calibration_vdf_dlink_results],sort=False)
            output_df = pd.concat([output_df,internal_vdf_link_df],sort=False)
            per_error=np.mean(np.abs(internal_vdf_link_df['vol_period_hourly']-internal_vdf_link_df['vol_valid'])/internal_vdf_link_df['vol_valid'])
            per_error_demand=np.mean(np.abs(internal_vdf_link_df['vol_period']-internal_vdf_link_df['demand_est'])/internal_vdf_link_df['demand_est'])
            alpha_1=np.mean(internal_vdf_link_df.alpha)
            beta_1=np.mean(internal_vdf_link_df.beta)
            peak_factor_1=np.mean(internal_vdf_link_df.peak_hour_factor)
            para=[vdf_name,100*vdf_name[1]+vdf_name[0],vdf_name[0],vdf_name[1],period_name,CUT_OFF_SPD,ULT_CAP,K_CRI,FFS,mm,peak_factor_1,alpha_1,beta_1,\
                per_error,per_error_demand]
            g_parameter_list.append(para)
            # para=[vdf_name,vdf_name[0], vdf_name[1],period_name, CUT_OFF_SPD,ULT_CAP,K_CRI,FFS,mm]
            # g_parameter_list.append(para)
            iter = iter + 1
            end_time=time.time()
            print('CPU time:',end_time-start_time,'s\n')
            if LOG_FILE ==1:
                log_file.write ('CPU time:'+str(end_time-start_time)+'s\n\n')

        # Step 4 Calibrate daily VDF function 
        print('Step 4: Calibrate daily VDF function for VDF_type:'+str(vdf_name)+'...\n')
        
        if LOG_FILE ==1:
            log_file.write('Step 4: Calibrate daily VDF function for VDF_type:'+str(vdf_name)+'...\n')
        
        start_time=time.time()

        temp_daily_df=temp_daily_df.reset_index(drop=True)
        temp_daily_df['VDF_TYPE']=100*temp_daily_df.AT+temp_daily_df.FT
        alpha_dict, beta_dict, temp_daily_df = vdf_calculation_daily(temp_daily_df, vdf_name, CUT_OFF_SPD, ULT_CAP,
                                                                     K_CRI, FFS, mm, ASSIGNMENT_PERIOD,
                                                                     UPPER_BOUND_DOC_RATIO, PERIOD_LENGTH)
        output_df_daily = pd.concat([output_df_daily,temp_daily_df],sort=False)
        
        end_time=time.time()
        print('CPU time:',end_time-start_time,'s\n')        
        
        if LOG_FILE ==1:
            log_file.write('CPU time:'+str(end_time-start_time)+'s\n\n')


    # Step 6 Output results 
    print('Step 5: Output...\n')
    if LOG_FILE ==1: 
        log_file.write('Step 5: Output...\n')
    para_df= pd.DataFrame(g_parameter_list)
    para_df.rename(columns={0:'VDF',
                                1:'VDF_TYPE',
                                2: 'FT',
                                3: 'AT',
                                4: 'period',
                                5:'CUT_OFF_SPD',
                                6:'ULT_CAP',
                                7:'K_CRI',
                                8:'FFS',
                                9: 'mm',
                                10:'peak_hour_factor',
                                11:'alpha',
                                12:'beta',
                                13:'per_error',
                                14:'per_error_demand'}, inplace=True)
    para_df.to_csv('./3_summary.csv',index=False)
    output_df_daily.to_csv('./3_calibration_daily_output.csv',index=False)
    output_df.to_csv('./3_calibration_output.csv',index=False)
    

    print('END...')
    if LOG_FILE ==1: 
        log_file.write('END...\n')
        log_file.close()