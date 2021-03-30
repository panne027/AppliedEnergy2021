s# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 08:19:49 2020

@author: panne027
"""

#%% Load Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, leastsq
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, Lasso, MultiTaskLasso, ElasticNet, MultiTaskElasticNet, Lars, OrthogonalMatchingPursuit, LassoLars
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, IsolationForest, RandomTreesEmbedding, StackingRegressor, VotingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
# import xgboost as xg 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
# import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
#%% Read Data
path='C:/Users/panne027/Google Drive/NOx Model3/400000DataDrayage_DWC.csv'
# path='D:/Google Drive/NOx Model3/200000DataRefuse_DWC.csv'
print(path)
# OBD1=pd.read_csv(path, usecols=["IntakeT","IntakekPa","engrpm","Fuelconskgph","EGRkgph","Airinkgph","SCRinppm","EngTq", 'Wheelspeed', 'Engpwr', 'RailMPa', 'SCRingps','NOxActual',"Anomalyindex3","Tadiab","tinj", 'NOxTheoryppmM3','EngTq', 'ExhaustT', 'EnginT','SCRoutgps', 'accelpedalpos'])

# OBD1=pd.read_csv('D:/Google Drive/NOx Model3/500000DataDrayage.csv', usecols=['time',"EngIntakeManifold1Temp","EngTurboBoostPress","EngSpeed","EngFuelRate","EngExhstGsRcrcltionMassFlowRate",
# OBD1=pd.read_csv(path, usecols=['time',"EngTurbo1CompressorIntakeTemp","EngTurbo1CompressorIntakePress","TransClutch_ConverterInputSpeed","EngFuelRate","EngExhstGsRcrcltionMassFlowRate","EngInletAirMassFlowRate", 'EngInjectorMeteringRail1Press','Aftertreatment1IntakeNOx', 'WheelBasedVehicleSpeed'])
                                                                                  
# not yard tractor:
# EngIntakeManifold1Temp
# OBD1=pd.read_csv(path, usecols=["EngIntakeManifold1Temp","EngTurboBoostPress","EngSpeed","EngFuelRate","EngExhstGsRcrcltionMassFlowRate","EngInletAirMassFlowRate", 'EngFuelDeliveryPress','Aftertreatment1IntakeNOx','WheelBasedVehicleSpeed'])
# OBD=pd.read_csv(path, usecols=["IntakeT","IntakekPa","engrpm","Fuelconskgph","EGRkgph","Airinkgph","SCRinppm","EngTq", 'Wheelspeed', 'Engpwr', 'RailMPa', 'SCRingps','NOxActual',"Anomalyindex3","Tadiab","tinj", 'NOxTheoryppmM3','EngTq', 'ExhaustT', 'EnginT','SCRoutgps', 'accelpedalpos'])
OBD1=pd.read_csv(path)
# OBD1=OBD1[OBD1['NOxActuallead']>=0]
# OBD1=OBD1[OBD1['y']!=0]
# OBD1.replace([np.inf, -np.inf], np.nan).dropna()
# OBD1[OBD1['KineticIntensity']!='#NAME?']=
OBD=OBD1
# 200000DataYardTractor
# 500000DataDrayage
# 100000DataFDT_DWC2
# 200000DataRefuse
# 160000DataFoodDelivery
# 100000Data

# 100000DataFoodDelivery_DWC2
# 200000DataYardTractor_DWC
# 400000DataDrayage_DWC
# 200000DataRefuse_DWC

# Combustion chamber Parameters****
S = 0.169#stroke (m)
B = 0.137#bore (m)
# S=0.124 
# B=0.107
# Yard Tractor Cummins QSB - 6.7: S=0.124 B=0.107
# Drayage Peterbilt 379 ISX15 (14.9L): S = 0.169 B = 0.137
# Food Delivery PX-7 (6.7L): S=0.124 B=0.107
# Refuse Cummins ISL: S=0.145 B=0.114

# OBD1=OBD1[OBD1['Aftertreatment1IntakeNOx']>0]
# OBD1=OBD1[OBD1['Aftertreatment1IntakeNOx']<3000]
# OBD1.replace([np.inf, -np.inf], np.nan).dropna()
# OBD1=OBD1[OBD1['EngSpeed']!=0]
# OBD1=OBD1[OBD1['EngFuelRate']!=0]
# OBD=OBD1[OBD1['EngIntakeManifold1Temp']!=0]
# OBD=OBD1[OBD1['EngExhstGsRcrcltionMassFlowRate']!=0]

# EngTurbo1CompressorIntakePress
# OBD=OBD1[OBD1['EngExhstGsRcrcltionMassFlowRate']!=0]

#EngTurboBoostPress
# T1=OBD.Wheelspeed
# gt100= OBD[OBD['Wheelspeed'].gt(100)].index
# print(gt100)

# T=OBD.iloc[51937:51937+60]
# T=OBD.iloc[7492:7492+60]
# T=OBD.iloc[1035:1035+60]
T=OBD[:]
# T=OBD[:100000]

# time=T['time'].to_numpy()
# Train=OBD.iloc[:10000]
# EnginT=T['EnginT'].to_numpy()
# ExhaustT=T['ExhaustT'].to_numpy()
# intakePlistlag=T['EngTurboBoostPress'].to_numpy() + 101325

# intakePlist=[]
# intakePlist[:]=intakePlistlag[1:]
# intakePlist=np.append(intakePlist, intakePlistlag[len(intakePlistlag)-1]) #Only for yard tractor

# 
# 
# +101325

# Fuelconskgphlag=T['EngFuelRate'].to_numpy()*0.835 #l/h * density = kg/h

# Fuelconskgph=[]
# Fuelconskgph=np.append(Fuelconskgph, Fuelconskgphlag[0]) #Only for yard tractor
# Fuelconskgph=np.append(Fuelconskgph, Fuelconskgphlag[:-1]) #Only for yard tractor
# Fuelconskgph[1:]=Fuelconskgphlag[:-2]

Fuelconskgph=T['EngFuelRate'].to_numpy()*0.835 #l/h * density = kg/h
EGRkgph=T['EngExhstGsRcrcltionMassFlowRate'].to_numpy()
Airinkgph=T['EngInletAirMassFlowRate'].to_numpy()
NOxActual=T['Aftertreatment1IntakeNOx'].to_numpy()
Wheelspeed=T['WheelBasedVehicleSpeed'].to_numpy()
RailMPa=T['EngFuelDeliveryPress'].to_numpy() #kPa to MPa
intakeTlist=T['EngIntakeManifold1Temp'].to_numpy()+273
intakePlist=T['EngTurboBoostPress'].to_numpy()*1000+101325 
intakeRPMlist=T['EngSpeed'].to_numpy()

# Fuelconskgph[Fuelconskgph==0]=0.0000001
# EGRkgph[EGRkgph==0]=0.0000001
# Airinkgph[Airinkgph==0]=0.0000001
# intakeRPMlist[intakeRPMlist==0]=0.0000001
# intakeTlist[intakeTlist==0]=0.0000001
# intakePlist[intakePlist==0]=0.0000001

# Model variables
X1=T['X1'].to_numpy()
X2=T['X2'].to_numpy()
X3=T['X3'].to_numpy()
X4=T['X4'].to_numpy()
y=T['y'].to_numpy()


# OBD['gps_Acc']=OBD['gps_Speed'].diff()/OBD['gps_Time'].diff()
# OBD['KineticIntensity']=OBD['gps_Acc']/OBD['gps_Speed']**2
# OBD['EngPwr']=OBD['ActMaxAvailEngPercentTorque']/100*OBD['EngReferenceTorque']*0.73756214927727/5252*OBD['EngSpeed']
# # 

# EngTq=T['EngTq'].to_numpy()
# Engpwr=T['Engpwr'].to_numpy()

# NOxActual=T['NOxActual'].to_numpy()
# SCRingps=T['SCRingps'].to_numpy()
# SCRoutgps=T['SCRoutgps'].to_numpy()
# SCRreduce=SCRingps-SCRoutgps
# xNOxActual=T['xNOxActual'].to_numpy()
# Anomalyindex=T['Anomalyindex3'].to_numpy()
# Tadiab=T['TadiabModel3'].to_numpy()
# trespercent=T['tres/tcycle'].to_numpy()
# xo2intake=T['xo2intake'].to_numpy()
# NOxTheoryppm=T['NOxTheoryppmM3'].to_numpy()
# EngTq=T['EngTq'].to_numpy()
# index=np.where(NOxActual<0)
# index1=np.where(NOxActual==3076.75)
# time.pop(index,index1)



a = 0.5*S#crank radius (m) S = 2a
l = 3.5*a#connecting rod length (m)  l/a = 3-4 for small and medium size engine

Vdis = (np.pi*B**2)/4 * S;  #cylinder displacement (m^3)6 6.7L

cr = 17.3#compression ratio
Ru = 8.31434# Gas constant J/(mole K)
Vc = Vdis/(cr-1)#clearance volume (m^3)
gamma=1.35#polytropic ratio/ specific heat ratio
LHV=42.64e6# Lower Heating Value J/kg
nc=0.9; #Combustion Efficiency

IVO=-9 #Degrees ATDC *****

MW_fuel = 0.19065#kg/mol
MW_O2 = 0.032#kg/mol
MW_product = 0.02885#kg/mol
MW_air = 0.029#kg/mol
thetap=3 #premixed combustion duration degrees

#%% Write data
# OBD['NOxActual']=NOxActual
# OBD['NOxActuallead']=NOxActuallead
# OBD['xNOxActual']=xNOxActual
# OBD['xNOxActuallead']=xNOxActuallead
OBD['X1']=X1
OBD['X2']=X2
OBD['X3']=X3
OBD['X4']=X4
OBD['y']=y
# path='C:/Users/panne027/Google Drive/NOx Model3/100000DataFoodDelivery_DWC2.csv'
path='D:/Google Drive/NOx Model3/200000DataYardTractor_DWC.csv'

OBD.to_csv(path, index=False)
#%% Calculations

Vivo= (2*a*np.cos(IVO)+np.sqrt(4*a**2*np.cos(IVO)**2-4*(a**2-l**2)))/2*np.pi*B**2/4
Tpeak=intakeTlist*(Vivo/Vc)**(gamma-1)
Ppeak= (Tpeak/intakeTlist)**(gamma/(gamma-1))*intakePlist

eqratio= 14.37/Airinkgph*Fuelconskgph
phinox=0.90

#intake oxygen concentration
xfuel= Fuelconskgph/(Airinkgph+EGRkgph+Fuelconskgph)
Exhaustkgph=Airinkgph+Fuelconskgph
xexh=abs((0.21*Airinkgph-636.68/190.65*Fuelconskgph)/(Exhaustkgph-EGRkgph))
xexhn2=0.79*Airinkgph/(Exhaustkgph+EGRkgph)
xexhco2=13.883*0.044*Fuelconskgph/(Exhaustkgph+EGRkgph)
xexhh2o=12.026*0.018*Fuelconskgph/(Exhaustkgph+EGRkgph)
xo2 = ((0.21*Airinkgph+xexh*EGRkgph)/(Airinkgph+EGRkgph+Fuelconskgph))
xo2intake=(19.90/phinox+xexh*EGRkgph/0.032/Fuelconskgph*MW_fuel)/(1+19.90/phinox*4.76+EGRkgph/Fuelconskgph*(xexh/0.032+xexhn2/0.028+xexhco2/0.044+xexhh2o/0.018))
xn2 = abs(1-xo2intake-xfuel)

# Injection duration equal to duration of diffusion combustion
# fuelinjrate=0.86*7*np.pi*0.00018**2/4*np.sqrt(2*872*(RailMPa*10**6-intakePlist)) #7 holes 0.007" diameter
# fuelinjrate=Fuelconskgph*2/(6*intakeRPMlist*60)
# tinj=Fuelconskgph/(fuelinjrate*30*intakeRPMlist)
tinj=Fuelconskgph*2/(6*60*113/1000000*835*intakeRPMlist) 
thetainj=tinj*intakeRPMlist/60*360
Ea=618840/(50+25)
mps=2*S*intakeRPMlist/60

thetaID=(0.36+0.22*mps)*np.exp(Ea*(1/(Ru*intakeTlist*cr**(gamma-1))-1/17190)+(21.2/(intakePlist*cr**gamma-12.4)**0.63))
# thetaID=3.45*np.exp(2100/intakeTlist)*(intakePlist)**(-1.02)/1000*intakeRPMlist/60*360
beta=0.45*thetaID/Fuelconskgph/1000000*intakeRPMlist*60*6 #On the Premixed Combustion in a Direct-Injection Diesel Engine 
# beta=0.5/thetainj
# tprem
tres=abs(tinj*(1-beta)) # residence time
# tres=tinj

#Adiabatic Flame Temperature
Tadiab=np.array([])
Nn2=np.array([])
No2=np.array([])
No2react=np.array([])

for i in range(len(NOxActual)):
    Energytotal=LHV*MW_fuel
    # Nair= 14.37*190.649/(eqratio[i]*28.84)
    Nn2= np.append(Nn2, xn2[i]*(EGRkgph[i]+Airinkgph[i]+Fuelconskgph[i])*MW_fuel/(0.028*Fuelconskgph[i]))
    # No2react=np.append(No2react, xo2intake[i]*(EGRkgph[i]+Airinkgph[i]+Fuelconskgph[i])*MW_fuel/(0.032*Fuelconskgph[i]))
    No2=np.append(No2, xo2[i]*(EGRkgph[i]+Airinkgph[i]+Fuelconskgph[i])*MW_fuel/(0.032*Fuelconskgph[i])-(13.883+6.013))
    a1Tadiab=(13.883*0.04453e2 + 12.026*0.02672e2 + Nn2[i]*0.02927e2 + No2[i]*0.03698e2)*Ru
    # a1Tadiab=(13.883*0.04453e2 + 12.026*0.02672e2 + Nn2[i]*0.02927e2 + No2[i]*0.03698e2)*Ru

    a2Tadiab2= 0.5*(13.883*0.03140e-1 + 12.026*0.03056e-1 + Nn2[i]*0.1488e-2 + No2[i]*0.06145e-2)*Ru
    coeff=[a2Tadiab2, a1Tadiab, -Energytotal]
    Tadiab=np.append(Tadiab, np.roots(coeff)[1]+Tpeak[i])

# xNOxActual calc and lead
# Air_in_col = np.array(data['EngInletAirMassFlowRate'], dtype = 'float') #kg/h
# m_dot = np.add(Air_in_col,(np.array(df['EngFuelRate'],dtype=float)*0.85))/3.6 #mass flow rate g/s
# g_nox_array = np.multiply(nox_array,m_dot)*1.587/1000000
# SCRingps= NOxActual*Fuelconskgph/3.6*1.587/100000
# xNOxActual=SCRingps/1000*3600/Exhaustkgph
xNOxActual=NOxActual/1000000
# dxNOxActual=xNOxActual[1:]-xNOxActual[:-1]
# EINOxActual=SCRingps/(Fuelconskgph/3600)

NOxActuallead=[]
NOxActuallead[:]=NOxActual[1:]
NOxActuallead=np.append(NOxActuallead, NOxActual[len(NOxActual)-1])
# NOxActuallead=np.append(NOxActuallead, NOxActual[len(NOxActual)-1])

# EINOxActuallead=[]
# EINOxActuallead[:]=EINOxActual[1:]
# EINOxActuallead=np.append(EINOxActuallead, EINOxActual[len(EINOxActual)-1])

xNOxActuallead=[]
xNOxActuallead[:]=xNOxActual[1:]
xNOxActuallead=np.append(xNOxActuallead, xNOxActual[len(xNOxActual)-1])

# tcycle=1/intakeRPMlist*60*2

#%% Curve fit
X1 = Tadiab
X2 = tres/intakeRPMlist*6
X3= abs(xo2intake)*(Airinkgph+EGRkgph)/3600/0.032

#%% Curve fit
X1 = Tadiab[:]
X2 = tres[:]*intakeRPMlist[:]
X3= abs(xo2intake[:])
X4=intakeTlist[:]

y=xNOxActuallead[:]

#%% Curve fit
X1 = Tadiab[:]
X2 = tres[:]*intakeRPMlist[:]
X3= abs(xo2intake[:])
y=np.log(xNOxActuallead[:])
#%% split
X=[]
X=np.array([np.log(X2), np.log(X3), np.log(X1/X4), -1/X1]).transpose()
# X_set1=np.array([intakeRPMlist, EGRkgph, Airinkgph, intakeTlist, intakePlist, Fuelconskgph]).transpose()
# X_set2=np.array([np.diff(intakeRPMlist), np.diff(EGRkgph), np.diff(Airinkgph), np.diff(intakeTlist), np.diff(intakePlist), np.diff(Fuelconskgph)]).transpose()
# X_set3=np.array([accelpedalpos,ExhaustT,Wheelspeed,EngTq,Instfuelecon]).transpose()
# X_set4=np.array([np.diff(accelpedalpos),np.diff(ExhaustT),np.diff(Wheelspeed),np.diff(EngTq),np.diff(Instfuelecon)]).transpose()
x_train, x_test, y_train, y_test=train_test_split(X,np.log(y),test_size=0.2)
# x_train, x_test, y_train, y_test=train_test_split(X_set3,np.log(y),test_size=0.2)
# x_train, x_test, y_train, y_test=train_test_split(X_set2,np.diff(np.log(y[:])),test_size=0.2)
# x_train, x_test, y_train, y_test=train_test_split(X_set4,np.diff(np.log(y[:])),test_si'ze=0.2)

# scaler = MinMaxScaler()
# X1_norm = scaler.fit_transform(X_set1)
#%% curvefit function

def xNOxpredict(X_NLR, a,b,c,d,e):
# def xNOxpredict(X, a):

    X1, X2, X3, X4 = X_NLR
    # return a+b*X1+c*X2
    return a*(X2**b)*(X3**c)*((X1/X4)**(d))*np.exp(-e/X1)
# 
    # return a+b*np.log(X2)+c*np.log(X3)+d*np.log(X1)-e/X1
    # return 5.13937125e+04*(X2**2.81034899e-01)*(X3**4.76560839e-01)*(X1**(-1.56412462e+00))*np.exp(-3.65674659e+03/X1)

X_NLR = X1, X2, X3, X4

# y=dxNOxActual[:]
# p0 = 5, 0.2, 0.5, -1.5 ,3000
# p0 = 200, 0.5, 0.5, -1.5, 3000 
p0= 20,0.5, 0.5,0.5,3000

# p0= 50000
fitParams, fitCovar= curve_fit(xNOxpredict, X_NLR, y, p0, maxfev=5000000, absolute_sigma=True, method='trf')

# fitParams=[ 1.38767819e-01, -9.08221893e-04,  3.69561515e-01, -1.41199476e+00, 1.50126139e+01] #TB Tadiab/intakeT
# [ 1.1501014   0.18922848  1.17569559 -1.61897682 14.13151179] #FDT

# fitParams=[2.15253326e-03, 1.33236679e-02, 3.59730214e-01, 8.16857902e-02, 2.11733407e+03] #TB intakeTlist
# [1.08274018e-03 1.88907008e-01 1.00399794e+00 8.20768972e-01 3.17874520e+02] #FDT
# fitParams=[7.46179642e-04, 2.35374116e-01, 7.42822494e-01, 8.20220527e-01, 7.32511216e+02] #FDT EGR=0 included

# ),bounds=((-np.inf, -np.inf, -np.inf,-np.inf, 0),(np.inf, np.inf, np.inf,np.inf,np.inf))
# ,bounds=((0, -np.inf, -np.inf,-np.inf, 0),(np.inf, np.inf, np.inf,np.inf,np.inf))
# , bounds=((-np.inf, -np.inf, -np.inf, 30000),(np.inf, np.inf, np.inf, 50000))
 # , bounds=((-np.inf, -np.inf, -np.inf, 0),(np.inf, np.inf, np.inf, np.inf)

print(fitParams)
print(fitCovar)
xNOxTheory=[]
xNOxTheory = xNOxpredict(X_NLR,fitParams[0], fitParams[1],fitParams[2], fitParams[3], fitParams[4])

#%%
NOxTheoryppm=[]
# NOxTheoryppm = xNOxTheory*1000000
NOxTheoryppm = np.exp(xNOxTheory)*1000000

NOxTheoryppmlag=np.array([])
NOxTheoryppmlag=np.append(NOxTheoryppmlag,NOxActual[0])
NOxTheoryppmlag=np.append(NOxTheoryppmlag,NOxTheoryppm[:-1])
# Error=NOxTheoryppmlag-NOxActual[:] 
# AbsError= np.abs(NOxTheoryppmlag-NOxActual[:])
# NOxTheoryppmlag=np.array(NOxTheoryppmlag)
# SCRingpsTheory= EINOxTheory/ 3600 * Fuelconskgph[:]

# Metrics
r_value= scipy.stats.linregress(NOxActual[:], NOxTheoryppmlag[:])
R2value=r_value[2]**2
mae=mean_absolute_error(NOxActual[:], NOxTheoryppmlag[:])
rmse=np.sqrt(mean_squared_error(NOxActual[:], NOxTheoryppmlag[:]))
print('[R2,RMSE,MAE]= [{} {} {}]'.format(R2value,rmse,mae))

plt.rcParams.update({'font.size': 70})
plt.figure(figsize=(35,30))
plt.scatter(NOxActual[:], NOxTheoryppmlag[:], c=abs(NOxTheoryppmlag[:]-NOxActual[:]), cmap='viridis' )
plt.plot(range(0,int(2000), 1),range(0,int(2000), 1))
plt.axis('square')
plt.xlim(0, 800)
plt.ylim(0, 800)
# NOxTheoryppm=2.6e7*((tres/intakeRPMlist*6)**1)*(xo2**0.5)*(Tadiab**0)*np.exp(-1100/Tadiab)
title='NOx Prediction RFR-TB pred X '
# Yard Tractor, Food Delivery, Refuse, Transit Bus cross_val_pred paramset1
# n_estimators=25,max_depth=25, minsamp=500
# test_size=0.3, n_est=25, depth=20, min_split=500
# plt.title(title,fontsize=60)
# plt.title('Training data Neural Network',fontsize=45)
plt.xlabel('NOx Observed /ppm')
plt.ylabel('NOx Predicted /ppm')
cbar=plt.colorbar()
plt.clim(0,1000)
cbar.set_label('Divergence /ppm')
print('Plot: {}'.format(title))

#%% RandomForestRegressor
# RandomForestRegressor, AdaBoostRegressor,BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomTreesEmbedding, StackingRegressor, VotingRegressor, HistGradientBoostingRegressor, xg.XGBRegressor
reg5=RandomForestRegressor(n_estimators=5, max_depth=3, min_samples_split=300, criterion='mse').fit(x_train,y_train)
# print(reg4.intercept_, reg4.coef_)n_estimators=25, max_depth=20, min_samples_split=15
#
# Xpred=[]

xNOxTheory=[]
xNOxTheory=reg5.predict(X)
#%%
# Xpred=reg5.predict(X_set1)
r_value= scipy.stats.linregress(np.diff(np.log(y)), xNOxTheory[:])
R2value=r_value[2]**2
mae=mean_absolute_error(np.diff(np.log(y)), xNOxTheory[:])
rmse=np.sqrt(mean_squared_error(np.diff(np.log(y)), xNOxTheory[:]))
print('[R2,RMSE,MAE]= [{} {} {}]'.format(R2value,rmse,mae))
featureimportance=reg5.feature_importances_
print(featureimportance)
#%%
subestimators=reg5.estimators_
tree0=reg5.base_estimator_
tree01=subestimators[0].tree_
#%%
featureimportance = np.mean([
    tree.feature_importances_ for tree in reg5.estimators_
], axis=0)
#%%
feat_imp_dict = reg5.get_booster().get_score(importance_type='gain')
featureimportance = np.asarray([feat_imp_dict.get(i, 0) for i in self.features])
#%%
featureimportance=reg5.feature_importances_
#%%
from rfpimp import permutation_importances

def r2(reg5, x_train, y_train):
    return r2_score(y_train, reg5.predict(x_train))

perm_imp_rfpimp = permutation_importances(reg5, x_train, y_train, r2)
#%% k fold cross validation

cv = KFold(n_splits=10, random_state=0, shuffle=True)
modelRF=RandomForestRegressor(n_estimators=25, max_depth=20, min_samples_split=10)
# modelNLR=curve_fit(xNOxpredict, X, y, p0, maxfev=5000000, absolute_sigma=True, method='trf')
scores = cross_val_score(modelRF, X_set1, np.diff(np.log(y)), cv=cv, scoring='r2')
# np.log(y)
#%% 
from geopy.distance import distance, geodesic
from geopy import Point
# coords_1=((OBD['gps_Lat'][:-1], OBD['gps_Long'][:-1]))
# coords_2=list((OBD['gps_Lat'][1:], OBD['gps_Long'][1:]))
# OBD['point'] = OBD['gps_Lat'].astype(float),OBD['gps_Long'].astype(float)
OBD['point'] = OBD.apply(lambda row: Point(latitude=row['gps_Lat'], longitude=row['gps_Long']), axis=1)
OBD['point_next'] = OBD['point'].shift(1)
OBD.loc[OBD['point_next'].isna(), 'point_next'] = None
OBD['distance_km'] = OBD.apply(lambda row: geodesic(row['point'], row['point_next']).km if row['point_next'] is not None else float('nan'), axis=1)
OBD = OBD.drop('point_next', axis=1)    

#%% Kinetic Intensity
acc=0
v2aero=0
notindex=OBD.index[OBD['distance_km']>=0.3].tolist()
for i in range(2,len(OBD['gps_Speed'])):
    if OBD['pass'][i]==OBD['pass'][i-1] and i not in notindex:
        acc+=abs(0.5*((OBD['gps_Speed'][i]/3.6)**2-(OBD['gps_Speed'][i-1]/3.6)**2)+9.81*(OBD['distance_km'][i])*1000)
        v2aero+=(OBD['gps_Speed'][i]/3.6)**3*(OBD['gps_Time'][i]-OBD['gps_Time'][i-1])
ki=acc/v2aero
# HghRslutionTotalVehicleDistance
print('Kinetic Intensity= {}'.format(ki))

#%% cross_val_predict
modelRF=RandomForestRegressor(n_estimators=25, max_depth=20, min_samples_split=15)
cv = KFold(n_splits=10, random_state=0, shuffle=True)

xNOxTheory=[]
xNOxTheory=cross_val_predict(modelRF, X_set4, np.diff(np.log(y[:])), cv=cv)
#%%
xNOxTheory_train=np.exp(reg5.predict(x_train))*1000000
xNOxTheory_test=np.exp(reg5.predict(x_test))*1000000


r_value= scipy.stats.linregress(np.exp(y_train[:])*1000000, xNOxTheory_train[:])
R2value=r_value[2]**2
mae=mean_absolute_error(np.exp(y_train[:])*1000000, xNOxTheory_train[:])
rmse=np.sqrt(mean_squared_error(np.exp(y_train[:])*1000000, xNOxTheory_train[:]))
print('Train [R2,RMSE,MAE]= [{} {} {}]'.format(R2value,rmse,mae))

r_value= scipy.stats.linregress(np.exp(y_test[:])*1000000, xNOxTheory_test[:])
R2value=r_value[2]**2
mae=mean_absolute_error(np.exp(y_test[:])*1000000, xNOxTheory_test[:])
rmse=np.sqrt(mean_squared_error(np.exp(y_test[:])*1000000, xNOxTheory_test[:]))
print('Test [R2,RMSE,MAE]= [{} {} {}]'.format(R2value,rmse,mae))
#%%
fn=['X1','X2','X3','X4']
fn=['log(t_res/engRPM)', 'log(x_O2)', 'log(T_adiab/intakeT)', '-1/T_adiab']
cn=['log(xNOx)']
#%%
# from sklearn.tree import export_graphviz
fig, axes = plt.subplots(nrows = 1, ncols = 1,figsize = (50,20), dpi=400 )
plot_tree(subestimators[0],
               feature_names = fn,
               class_names=cn,
               fontsize=30,
               filled = True, rounded=True,proportion=True
               );
fig.savefig(path+'rf_individualtree3.png')
plt.show()

#%%
plt.rcParams.update({'font.size': 60})
plt.figure(figsize=(60,40))
n_classes = 1
plot_colors = "ryb"

for pairidx, pair in enumerate([ [0, 1], [0, 2],[2,1], [3, 1], [3,0],[3,2]]):
    # We only take the two corresponding features
    X_db = X[:, pair]
    y_db = np.log(y)

    # Train
    # clf = DecisionTreeRegressor().fit(X_db, y_db)
    clf=tree0.fit(X_db[:],y_db)
    # Plot the decision boundary
    plt.subplot(2,3, pairidx + 1)
    # plt.axis('equal')
    plot_stepx =abs(X_db[:, 0].max()-X_db[:, 0].min())*0.001
    plot_stepy =abs(X_db[:,1].max()-X_db[:,1].min())*0.001
    x_min, x_max = X_db[:, 0].min()-plot_stepx, X_db[:, 0].max()+plot_stepx
    y_min, y_max = X_db[:,1].min()-plot_stepy, X_db[:,1].max()+plot_stepy
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_stepx),
                         np.arange(y_min, y_max, plot_stepy))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.xlabel(fn[pair[0]])
    plt.ylabel(fn[pair[1]])
    plt.scatter(X_db[:,0], X_db[:,1], label=cn[0], color=cs,
                    cmap=plt.cm.Spectral, s=4, alpha=0.8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # plt.axis("tight")
# plt.suptitle("Decision Surface of a decision tree")
# plt.legend(loc='lower right', borderpad=0, handletextpad=0)
# 

# plt.figure()
# clf = tree0.fit(X, np.log(y))
# plot_tree(clf, filled=True)
# plt.show()
    
#%%
from dtreeviz.trees import dtreeviz # remember to load the package

viz = dtreeviz(subestimators[0], x_data=X, y_data=np.log(y),
                target_name=cn[0],
                feature_names=fn, show_node_labels = True, )
viz
#%%

from sklearn.tree import export_graphviz
# Export as dot file
fig, axes = plt.subplots(nrows = 1, ncols = 1,figsize = (50,15), dpi=800 )
export_graphviz(subestimators[0], out_file=path+'tree.dot', 
                feature_names = fn,
                class_names = cn,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', path+'tree.dot', '-o', path+'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = path+'tree.png')

#%% Deep neural network

scaler = MinMaxScaler()

X_DNN= np.array([np.log(X1), np.log(X2), np.log(X3)]).transpose()
X_norm = scaler.fit_transform(X_DNN)

y_NN=np.log(y)
# reg4 = MLPRegressor(, max_iter=1000000).fit(x_train, y_train)
reg4 = MLPRegressor(hidden_layer_sizes=(100,),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
               learning_rate='constant', learning_rate_init= 0.01, power_t=0.5, max_iter=1000000, shuffle=True,
               random_state=10, tol=0.0001, verbose=False, warm_start=False,
                early_stopping=False, validation_fraction=0.1, beta_1=0.95, beta_2=0.999,momentum=0.95,
               epsilon=1e-08)
fit4=reg4.fit(X_norm,y_NN)

# print(reg4.intercepts_, reg4.coefs_)

# xNOxTheory=[]
# xNOxTheory=reg4.predict(X)
NOxTheoryppm=[]
NOxTheoryppm=np.exp(fit4.predict(X_norm))*1000000

#%% leastsq

def xNOxpredictres(X, coeffs, y):
# def xNOxpredict(X, a):
    
    # return a+b*X1+c*X2
    return (y-coeffs[0]*(X[1]**coeffs[1])*(X[2]**0.5)*(X[0]**(-0.5))*np.exp(-coeffs[2]/X[0]))
    # return 5.13937125e+04*(X2**2.81034899e-01)*(X3**4.76560839e-01)*(X1**(-1.56412462e+00))*np.exp(-3.65674659e+03/X1)
X=np.array([X1, X2, X3])
  
p0= np.array([5, 0.2 ,3000], dtype=float)

x,flag = leastsq(xNOxpredictres, p0, args=(y,X))

print(x)

#%%
NOxTheoryppm=xNOxTheory*1000000
SCRTheory= xNOxTheory/3.6*Exhaustkgph
NOxTheoryppm=SCRTheory/Fuelconskgph*3.6/1.587*1000000
#%%
NOxTheoryppm= xNOxpredict((X1[:],X2[:], X3[:]),fitParams[0], 1 ,0.5, 0,  fitParams[4])


#%% linear regression
reg1=LinearRegression().fit(x_train,y_train)
print(reg1.intercept_, reg1.coef_)
xNOxTheory=[]
xNOxTheory=reg1.predict(X)



#%% ridge
reg2=Ridge(alpha=2,max_iter=1000000).fit(x_train,y_train)
print(reg2.intercept_, reg2.coef_)
xNOxTheory=[]
xNOxTheory=reg2.predict(X)
#%% Bayesianridge
reg3=BayesianRidge(n_iter=1000000).fit(x_train,y_train)
print(reg3.intercept_, reg3.coef_)
xNOxTheory=[]
xNOxTheory=reg3.predict(X)

#%% RandomForestRegressor
reg4=RandomForestRegressor().fit(x_train,y_train)
# print(reg4.intercept_, reg4.coef_)
xNOxTheory=[]
xNOxTheory=reg4.predict(X)


#%% RandomForestRegressor1
reg4=RandomForestRegressor().fit(x_train,y_train)
# print(reg4.intercept_, reg4.coef_)
xNOxTheory=[]
xNOxTheory=reg4.predict(X)


# %% neural network data
X1nn=Tadiab
X2nn=tres
X3nn=xo2
# X4nn=EngTq

Xnn=np.array([X1nn, X2nn, X3nn, X4nn]).transpose()

x_trainnn, x_testnn, y_trainnn, y_testnn=train_test_split(Xnn,y,test_size=0.5)


#%% Tensorflow architecture
scaler = MinMaxScaler()

Xnn= scaler.fit_transform(np.array([np.log(X1), np.log(X2), np.log(X3)]).transpose())

x_trainnn, x_testnn, y_trainnn, y_testnn=train_test_split(Xnn,np.log(y),test_size=0.2)
#%%
input_layer = Input(shape=(Xnn.shape[1],))
dense_layer_1 = Dense(20, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(5, activation='relu')(dense_layer_2)
output = Dense(1)(dense_layer_3)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
history = model.fit(x_trainnn, y_trainnn, batch_size=32, epochs=30, verbose=1, validation_split=0.2)
#%%
NOxTheoryppm=np.exp(model.predict(Xnn))*1000000

#%% tres DNN
Xnn= scaler.fit_transform(np.array([Fuelconskgph[:], intakeRPMlist[:]]).transpose())

x_trainnn, x_testnn, y_trainnn, y_testnn=train_test_split(Xnn,tres,test_size=0.3)

input_layer = Input(shape=(Xnn.shape[1],))
dense_layer_1 = Dense(20, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(5, activation='relu')(dense_layer_2)
output = Dense(1)(dense_layer_3)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
history = model.fit(x_trainnn, y_trainnn, batch_size=2, epochs=20, verbose=1, validation_split=0.2)


#%% EINOx to NOxppm
NOxTheoryppm=[]
# molesNOx= EINOxTheory/38 * Fuelconskgph[:] #moles per hour
molesNOx= xNOxTheory*Exhaustkgph[:]/0.040*3
totalproductmoles= (Nn2[:] + No2[:] + 13.883 + 12.026)* Fuelconskgph[:]/MW_fuel #moles per hour
NOxTheoryppm= molesNOx/totalproductmoles*1e6

#%% Metrics
NOxTheoryppmlag=np.array([])
NOxTheoryppmlag=np.append(NOxTheoryppmlag,NOxActual[0])
NOxTheoryppmlag=np.append(NOxTheoryppmlag,NOxTheoryppm[:-1])
Error=NOxTheoryppmlag-NOxActual[:] 
AbsError= np.abs(NOxTheoryppmlag-NOxActual[:])
# NOxTheoryppmlag=np.array(NOxTheoryppmlag)
# SCRingpsTheory= EINOxTheory/ 3600 * Fuelconskgph[:]

# Metrics
r_value= scipy.stats.linregress(NOxActual[:], NOxTheoryppmlag[:])
R2value=r_value[2]**2
mae=mean_absolute_error(NOxActual[:], NOxTheoryppmlag[:])
rmse=np.sqrt(mean_squared_error(NOxActual[:], NOxTheoryppmlag[:]))
print('[R2,RMSE,MAE]= [{} {} {}]'.format(R2value,rmse,mae))

# r_value= scipy.stats.linregress(xNOxActuallead[:], xNOxTheory[:])
# R2value=r_value[2]**2
# mae=mean_absolute_error(xNOxActuallead[:], xNOxTheory[:])
# rmse=np.sqrt(mean_squared_error(xNOxActuallead[:], xNOxTheory[:]))
# print('[R2,RMSE,MAE]= [{} {} {}]'.format(R2value,rmse,mae))

#%% Baseline plot
plt.rcParams.update({'font.size': 60})
plt.figure(figsize=(35,40))
plt.scatter(NOxActual[:], NOxTheoryppmlag[:], c=abs(NOxTheoryppmlag[:]-NOxActual[:]), cmap='viridis' )
plt.plot(range(0,int(2000), 1),range(0,int(2000), 1))
plt.xlim(0, 800)
plt.ylim(0, 900)
# NOxTheoryppm=2.6e7*((tres/intakeRPMlist*6)**1)*(xo2**0.5)*(Tadiab**0)*np.exp(-1100/Tadiab)
title='NOx Prediction from curvefit for Drayage'
# Yard Tractor, Food Delivery, Refuse, Transit Bus
# plt.title(title,fontsize=60)
# plt.title('Training data Neural Network',fontsize=45)
plt.xlabel('NOx Observed /ppm', fontsize=60)
plt.ylabel('NOx Predicted /ppm', fontsize=60)
cbar=plt.colorbar()
plt.clim(0,1000)
cbar.set_label('Divergence /ppm', fontsize=60)
print('Plot: {}'.format(title))

#%% xNOx compare plot
plt.rcParams.update({'font.size': 45})
plt.figure(figsize=(35,35))
plt.scatter(SCRingps[:], SCRTheory[:], c=abs(SCRTheory[:]-SCRingps[:]), cmap='viridis' )
plt.plot(range(0,int(1), 1),range(0,int(1), 1))
plt.xlim(0, 0.1)
plt.ylim(0, 0.1)
# NOxTheoryppm=2.6e7*((tres/intakeRPMlist*6)**1)*(xo2**0.5)*(Tadiab**0)*np.exp(-1100/Tadiab)
plt.title('Yard Tractor NOxModel3 curvefit BoostPress SCR 100k points',fontsize=45)
# plt.title('Training data Neural Network',fontsize=45)
plt.xlabel('SCR Observed)', fontsize=60)
plt.ylabel('SCR Predicted', fontsize=60)
cbar=plt.colorbar()
# plt.clim(0,1000)
cbar.set_label('Divergence ', fontsize=60)
#%% histogram2d
# heatmap, xedges, yedges = np.histogram2d(NOxActual, NOxTheoryppmlag, bins=100)
# heatmap = gaussian_filter(heatmap, sigma=2)
extent = [0, 1000, 0, 1000] 
plt.rcParams.update({'font.size': 45})
plt.figure(figsize=(35,35))
# plt.show()

# cbar=plt.colorbar()
# plt.clim(0,1000)
# plt.clf()
# plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.plot(range(0,int(1000), 1),range(0,int(1000), 1))
plt.title('jointplot for proposed model',fontsize=45)
# plt.title('Training data Neural Network',fontsize=45)
plt.xlabel('NOx Observed/ppm)', fontsize=60)
plt.ylabel('NOx Predicted /ppm', fontsize=60)

#%% jointplot
comparedata=[]
h=sns.jointplot(x=NOxActual, y=NOxTheoryppmlag, height=40, xlim=(0,1000),ylim=(0,1000))
ax.set(xlabel='NOx Observed /ppm', ylabel='NOx Predicted /ppm')
h.set_axis_labels('NOx Observed /ppm', 'NOx Predicted /ppm')
h.fig.suptitle("Jointplot for NOxModel3 Prediction")
sns.lineplot(range(0,int(1000), 1),range(0,int(1000), 1))
h.ax_joint.plot(range(0,int(1000), 1),range(0,int(1000), 1), linewidth = 2)
#%% hexbin
plt.hexbin(x, y, c=z, gridsize=30, cmap=plt.cm.jet, bins=None)
plt.axis([x.min(), x.max(), y.min(), y.max()])

cb = PLT.colorbar()
cb.set_label('mean value')
PLT.show()   

#%% Engine load vs error plot
plt.rcParams.update({'font.size': 45})
plt.figure(figsize=(35,35))
plt.scatter(xo2intake[:], xNOxActual[:])
# plt.plot(np.linspace(0,0.1, 1),np.zeros(10))
# plt.xlim(0.0  , 0.015)
# plt.ylim(-750, 750)
plt.title('xo2 vs xNOxActual ',fontsize=45)
plt.xlabel('xo2 ', fontsize=45)
plt.ylabel('xNOxActual ', fontsize=45)
# cbar=plt.colorbar()
# # plt.clim(0,1000)
# # cbar.set_label('Divergence /ppm', fontsize=45)

#%%
plt.rcParams.update({'font.size': 45})
plt.figure(figsize=(50,10))
Time=np.arange(len(NOxActual))
plt.plot(Time,Wheelspeed)
plt.minorticks_on()
plt.grid(b=True, which='major', color='b', linestyle='-')



#%%
clusterindex=np.array(np.where(np.logical_and(NOxActual>125, NOxActual<150, )))
clusterindextheory=np.array(np.where(np.logical_and(NOxTheoryppm<490, NOxTheoryppm>400)))
clusterindex=np.intersect1d(clusterindex, clusterindextheory)
plt.subplots(4)
plt.rcParams.update({'font.size': 45})
time=np.arange(1,61)
plt.rcParams.update({'font.size': 45})
plt.style.use('seaborn-ticks')
fig1,ax1=plt.subplots(7, sharex=False, sharey=False, figsize=(30,75))

ax1[0].scatter(NOxActual[clusterindex], NOxTheoryppmlag[clusterindex] )
ax1[0].plot(range(0,int(1000), 1),range(0,int(1000), 1))
ax1[0].set_xlim(0, 1000)
ax1[0].set_ylim(0, 1000)
# NOxTheoryppm=2.6e7*((tres/intakeRPMlist*6)**1)*(xo2**0.5)*(Tadiab**0)*np.exp(-1100/Tadiab)
ax1[0].set_title('Cluster',fontsize=45)
# plt.title('Training data Neural Network',fontsize=45)
ax1[0].set_xlabel('NOx Observed/ppm)', fontsize=45)
ax1[0].set_ylabel('NOx Predicted /ppm', fontsize=45)

# ax1[0].set_title('Data Visualization')
# ax1[0].plot(time, xNOxActual[index])

# ax1[0].set_ylabel('$x_{NO_{x}}$ Observed')
# ax1[0].grid()
# ax1[0].minorticks_on()
ax1[1].scatter(clusterindex, intakeTlist[clusterindex])
ax1[1].set_ylabel('intakeTlist /K')
ax1[1].set_xlabel('Timestamp')
ax1[1].grid()
ax1[1].minorticks_on()
# ax1[2].scatter(clusterindex,Wheelspeed[clusterindex])
# ax1[2].set_ylabel('Wheelspeed')
# ax1[2].grid()
# ax1[2].minorticks_on()
# ax1[2].set_xlabel('Timestamp')

# ax1[3].scatter(clusterindex,EngTq[clusterindex])
# ax1[3].set_ylabel('Engine Load')
# ax1[3].grid()
# ax1[3].minorticks_on()
# ax1[3].set_xlabel('Timestamp')


ax1[4].scatter(clusterindex, intakeRPMlist[clusterindex])
ax1[4].set_ylabel('Engine RPM')
ax1[4].grid()
ax1[4].minorticks_on()
ax1[4].set_xlabel('Timestamp')

ax1[5].scatter(clusterindex, EGRkgph[clusterindex])
ax1[5].set_ylabel('EGRkgph')
ax1[5].grid()
ax1[5].minorticks_on()
ax1[5].set_xlabel('Timestamp')

ax1[6].scatter(clusterindex, SCRreduce[clusterindex])
ax1[6].set_ylabel('SCRingps-SCRoutgps')
ax1[6].grid()
ax1[6].minorticks_on()
ax1[6].set_xlabel('Timestamp')
#%%full plot
plt.rcParams.update({'font.size': 45})
plt.style.use('seaborn-ticks')
fig,ax=plt.subplots(8, sharex=False, sharey=False, figsize=(40,80))
time=np.arange(len(NOxActual))
ax[0].set_title('In Model parameters Transit Bus', fontsize=55)
ax[0].plot(time, intakeRPMlist)
ax[0].set_ylabel('intakeRPMlist', fontsize=45)
ax[0].grid()
ax[0].minorticks_on()

ax[1].plot(time, intakePlist)
ax[1].set_ylabel('intakePlist /Pa', fontsize=45)
ax[1].grid()
ax[1].minorticks_on()

ax[2].plot(time, intakeTlist)
ax[2].set_ylabel('intakeTlist /K', fontsize=45)
ax[2].grid()
ax[2].minorticks_on()

ax[3].plot(time, Fuelconskgph)
ax[3].set_ylabel('Fuelconskgph', fontsize=45)
ax[3].grid()
ax[3].minorticks_on()

ax[4].plot(time, Airinkgph)
ax[4].set_ylabel('Airinkgph', fontsize=45)
ax[4].grid()
ax[4].minorticks_on()

ax[5].plot(time, EGRkgph)
ax[5].set_ylabel('EGRkgph', fontsize=45)
ax[5].grid()
ax[5].minorticks_on()

ax[6].plot(time, RailMPa)
ax[6].set_ylabel('RailMPa', fontsize=45)
ax[6].grid()
ax[6].minorticks_on()

ax[7].plot(time, NOxActual)
ax[7].set_xlabel('time index', fontsize=45)
ax[7].set_ylabel('NOxActual', fontsize=45)
ax[7].grid()
ax[7].minorticks_on()


#%%
# index=np.arange(60000,60000+60)#Refuse
# index=np.arange(30000,30000+60)#Yard Tractor
# index=np.arange(62000,62000+60)#Drayage
# index=np.arange(27500,27500+60)#Food Delivery
point=5000
index=np.arange(point,point+60)
#%%
time=np.arange(1,61)
plt.rcParams.update({'font.size': 35})
plt.style.use('seaborn-ticks')
fig1,ax1=plt.subplots(8, sharex=True, sharey=False, figsize=(30,40))

ax1[0].set_title('Data Visualization Food Delivery {}'.format(point), fontsize=55)
ax1[0].plot(time, NOxActual[index], label='NOxActual')

ax1[0].set_ylabel('$NO_{x}$ ppm')
ax1[0].grid()
ax1[0].minorticks_on()
ax1[0].plot(time, NOxTheoryppmlag[index], label='NOxTheory with lag')
# ax1[0].set_ylabel('NOxTheory with lag')
ax1[0].legend()
ax1[1].plot(time, Wheelspeed[index])
ax1[1].set_ylabel('Wheelspeed')
ax1[1].grid()
ax1[1].minorticks_on()
# ax1[1].grid()
ax1[1].minorticks_on()
ax1[2].plot(time,Airinkgph[index])  
ax1[2].set_ylabel('Airinkgph')
ax1[2].grid()
ax1[2].minorticks_on()
ax1[3].plot(time,Fuelconskgph[index])
ax1[3].set_ylabel('Fuelconskgph')
ax1[3].grid()
ax1[3].minorticks_on()

ax1[4].plot(time, intakeRPMlist[index])
ax1[4].set_ylabel('Engine RPM')
ax1[4].grid()
ax1[4].minorticks_on()
ax1[5].plot(time, intakePlist[index])

ax1[5].set_ylabel('intakePlist')
ax1[5].grid()
ax1[5].minorticks_on()

ax1[6].plot(time, intakeTlist[index])
ax1[6].set_ylabel('intakeTlist')
ax1[6].grid()
ax1[6].minorticks_on()

ax1[7].plot(time,EGRkgph [index])
ax1[7].set_ylabel('EGRkgph')
ax1[7].grid()
ax1[7].minorticks_on()



# ax1[5].plot(time, EGRkgph[index])
# ax1[5].set_ylabel('EGRkgph')
# ax1[5].grid()
# ax1[5].minorticks_on()
# ax1[6].plot(time, NOxActual[index], label='NOx Actual ppm')
# ax1[6].set_ylabel('NOx Actual ppm')
# ax1[6].plot(time, NOxTheoryppmlag[index], label='NOx Predicted ppm')
# ax1[6].grid()
# ax1[6].legend()
# ax1[6].minorticks_on()
