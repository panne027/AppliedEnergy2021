# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:39:39 2020

@author: panne027
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
import random
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# 
# OBD=pd.read_csv('D:/Google Drive/NOx Model3/100000Data.csv', usecols=["IntakeT","IntakekPa","engrpm","Fuelconskgph","EGRkgph","Airinkgph","SCRinppm","EngTq", 'Wheelspeed', 'Engpwr', 'RailMPa', 'SCRingps','NOxActual', 'EINOxActual',"Anomalyindex5","Tadiab","tinj", 'NOxTheoryppmM3','nN2','nO2','pass'])
# OBD=pd.read_csv('C:/Users/panne027/Google Drive/NOx Model3/100000Data.csv',  usecols=["IntakeT","IntakekPa","engrpm","Fuelconskgph","EGRkgph","Airinkgph","SCRinppm","EngTq", 'Wheelspeed', 'Engpwr', 'RailMPa', 'SCRingps','NOxActual',"Anomalyindex5",'NOxTheoryppmM3'])
# # 
path='C:/Users/panne027/Google Drive/NOx Model3/100000DataFDT_DWC.csv'

OBD=pd.read_csv(path, usecols=["EngIntakeManifold1Temp","EngTurboBoostPress","EngSpeed","EngFuelRate","EngExhstGsRcrcltionMassFlowRate","EngInletAirMassFlowRate", 'EngFuelDeliveryPress','Aftertreatment1IntakeNOx', 'WheelBasedVehicleSpeed','NOxTheoryppm'])

# # gt100= OBD[OBD['Wheelspeed'].gt(100)].index
# # print(gt100)
# # T=OBD.iloc[51937:51937+60]
# # T=OBD.iloc[7492:7492+60]
# # T=OBD.iloc[1035:1035+60]
T=OBD

# intakeRPMlist=T['engrpm'].to_numpy()
# intakePlist=T['IntakekPa'].to_numpy()*1000
# intakeTlist=T['IntakeT'].to_numpy()+273
# Fuelconskgph=T['Fuelconskgph'].to_numpy()
# EGRkgph=T['EGRkgph'].to_numpy()
# Airinkgph=T['Airinkgph'].to_numpy()
# NOxActual=T['SCRinppm'].to_numpy()
# Wheelspeed=T['Wheelspeed'].to_numpy()
# EngTq=T['EngTq'].to_numpy()
# Engpwr=T['Engpwr'].to_numpy()
# RailMPa=T['RailMPa'].to_numpy()
# NOxTheoryppm=T['NOxTheoryppmM3'].to_numpy()
# SCRingps=T['SCRingps'].to_numpy()
# Error=NOxTheoryppm-NOxActual
# # xNOxActual=T['xNOxActual'].to_numpy()
# # Anomalyindex=T['Anomalyindex5'].to_numpy()
NOxTheoryppm=T['NOxTheoryppm'].to_numpy()
NOxActual=T['Aftertreatment1IntakeNOx'].to_numpy()


Error=NOxTheoryppm-NOxActual

# # xo2intake=T['xo2intake'].to_numpy()
# # Tadiab=T['Tadiab'].to_numpy()
# # trespercent=T['tres/tcycle'].to_numpy()
# NOxTheoryppm=T['NOxTheoryppmM3'].to_numpy()
# # Nn2=T['Nn2'].to_numpy()
# # No2=T['No2'].to_numpy()
# # passnumber=T['pass'].to_numpy()
# # 
# # passnumber=pd.Series(passnumber)


# Anomalies=pd.read_csv("D:/Google Drive/NOx Model3/Cooccurrence/windowpatterns/AnomalousWindows_99508_-1.0_10.0_3.csv")
# Anomalies=pd.read_csv("C:/Users/panne027/UMNME452/Prof.Northrop/Prof.Northrop/NOxModel3/AnomalousWindows_55255_-1.0_5.0_3.csv")
Anomalies=pd.read_csv("C:/Users/panne027/Google Drive/NOx Model3/Cooccurrence/FDT/AnomalousWindows_102370_-1.0_10.0_3.csv", header=None, names='1')
# Anomalies=pd.read_csv("D:/Google Drive/NOx Model3/Cooccurrence/windowpatterns/AnomalousWindows_99508_-1.0_10.0_3.csv", header=None, names='1')
Anomalyindex=Anomalies['1'].to_numpy()

# windows=pd.read_csv("D:/Google Drive/NOx Model3/Cooccurrence/windowpatterns/windowspatternskvaluepattern199508.0_7.0_0.004_60853.0_1.0_1.0.csv")
windows=pd.read_csv("C:/Users/panne027/Google Drive/NOx Model3/Cooccurrence/FDT/windowspatternskvaluepatternC6_3102370.0_6.0_0.004_82577.0_1.0_1.0.csv")
# windowspatterns=pd.read_csv("D:/Google Drive/NOx Model3/Cooccurrence/windowpatterns/windowspatterns2_99508.0_7.0_0.004_60853.0_1.0_1.0.csv", header=None, dtype=str, names='1')
windowspatterns=pd.read_csv("C:/Users/panne027/Google Drive/NOx Model3/Cooccurrence/FDT/windowspatterns2c6_3_102370.0_6.0_0.004_82577.0_1.0_1.0.csv", header=None, dtype=str,names='1')


windowlength=3
windows= windows.to_numpy()
windowspatterns=windowspatterns['1'].values.tolist()
params=pd.read_csv('C:/Users/panne027/Google Drive/NOx Model3/NOxModel3C7columns.csv')
# params=pd.read_csv('D:/Google Drive/NOx Model3/NOxModel3C7columns.csv')

params=params['Parameters'].to_numpy()
default="."
params=params+':'

overthreshold=0.5 #if 50% of windows in pattern underpredict, =1
# npatterns=1
#%% kvaluearray

over=[]
support=[]
ErrorAnomaly=[]
for j in Anomalyindex:
    error1=0
    for k in range(windowlength):
        if j+k< len(Error):
            error1+=Error[j+k]
    ErrorAnomaly.append(error1)
    if error1>=0:
        over.append(1)
    else:
        over.append(0)
totalanom=len(NOxActual)
#%%
kindex=[]
kvaluearray=[]
count=[]
support=[]
k=0
for i in range(len(windows)):
    if float(math.floor(windows[i])) != windows[i]:
        kindex.append(i)
        
        kvaluearray.append(float(windows[i]))
        if i==0:
            count.append(kindex[0])
            support.append(count[0]/totalanom)
        else:
            count.append(kindex[k]-kindex[k-1]-1)
            support.append(count[k]/totalanom)
        k+=1
#%%
kindexpatterns=[]
for i in range(len(windowspatterns)):
    try:
        float(windowspatterns[i])
        kindexpatterns.append(i)
    except ValueError:
        continue        
        

#%%
patterns=[]
minsupp=0.004
kvaluethreshold=1
kvaluethres=[]
countthres=[]
supportthres=[]
for i in range(1,len(kvaluearray)-1):
    if (kvaluearray[i]>=kvaluethreshold) and ((kindex[i]-kindex[i-1]-1)/len(NOxActual)>=minsupp):
        list1=[]
        k=0
        for k in range(kindex[i-1]+1,kindex[i+1],1):
            list1.append(int(windows[k]))
        patterns.append(list1)
        kvaluethres.append(kvaluearray[i])
        countthres.append(count[i])
        supportthres.append(support[i])
    else:
        continue
    #%%
patternover=[]
over=np.array(over,None)
overall=np.zeros(len(NOxActual))
# underall[:]=np.NaN
overall[Anomalyindex]=over
for list2 in patterns:
    if np.sum(overall[list2])/len(list2)>=overthreshold:
        patternover.append(1)
    else:
        patternover.append(0)
    
#%%

cooccpatterns=[np.full((len(params)),default).tolist() for _ in range(len(patterns))]
i=0
k=0
l=0
for l in range(len(support)-1):
    if (i<kindexpatterns[l]) and (kvaluearray[l]>=kvaluethreshold) and (support[l]>=minsupp):
        if i==0:
            for x in range(kindexpatterns[l]-0):
                j=np.where(params==windowspatterns[i].split('\t',1)[0])[0][0]
                # if(windowspatterns[i][0].split('\t',1)[0]==params[j]):
                cooccpatterns[k][j]=windowspatterns[i].split('\t',1)[1]
                print('i{}'.format(i))
                i+=1
                
                
        else:
            for x in range(kindexpatterns[l-1]+1,kindexpatterns[l]):
                j=np.where(params==windowspatterns[i].split('\t',1)[0])[0][0]
                # if(windowspatterns[i][0].split('\t',1)[0]==params[j]):
                cooccpatterns[k][j]=windowspatterns[i].split('\t',1)[1]
                print('i{}'.format(i))
                i+=1
                
        i+=1
        k+=1
    else:
        i=kindexpatterns[l]+1
    
    # print('L{}'.format(l))
#%%
topkindex=np.argsort(kvaluethres).tolist()
topkvalues=np.array(kvaluethres)[topkindex][::-1]
topkpatterns=np.array(cooccpatterns)[topkindex][::-1]
topksupport=np.array(supportthres)[topkindex][::-1]
topkcount=np.array(countthres)[topkindex][::-1]

#%%

#%%
path2='C:/Users/panne027/Google Drive/NOx Model3/Cooccurrence/'
# path2='D:/Google Drive/NOx Model3/Cooccurrence/'

patternspd=pd.DataFrame(cooccpatterns, columns=params)
patternspd.insert(len(params),'Support', supportthres, True)
patternspd.insert(len(params)+1,'Count', countthres, True)
patternspd.insert(len(params)+2,'Kvalue', kvaluethres, True)
patternspd.insert(len(params)+3,'IsOver',patternover , True)

patternspd.to_csv(path2+'NOxModel3DWCPatternsFDTC6_Underover.csv', index=False)
print(k)


           
























