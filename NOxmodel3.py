# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 23:33:18 2020

@author: phari
"""

#%% Read data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns
import random
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Activation, Dropout
# import seaborn as sns
# 
#%%
# path='D:/Google Drive/NOx Model3/100000Data.csv'
path='D:/Google Drive/NOx Model3/100000Data.csv'
# path='C:/Users/panne027/Google Drive/NOx Model3/100000Data.csv'

# path='C:/Users/panne027/Google Drive/NOx Model3/100000Data.csv'
# OBD=pd.read_csv(path, usecols=["IntakeT","IntakekPa","engrpm","Fuelconskgph","EGRkgph","Airinkgph","SCRinppm","EngTq", 'Wheelspeed', 'Engpwr', 'RailMPa', 'SCRingps','NOxActual',"Anomalyindex3","Tadiab","tinj", 'NOxTheoryppmM3','EngTq', 'ExhaustT', 'EnginT','SCRoutgps', 'accelpedalpos', 'Instfuelecon', 'ExhaustT', 'Wheelspeed'])
# OBD=pd.read_csv(, usecols=["IntakeT","IntakekPa","engrpm","Fuelconskgph","EGRkgph","Airinkgph","SCRinppm","EngTq", 'Wheelspeed', 'Engpwr', 'RailMPa', 'SCRingps','NOxActual',"Anomalyindex3","Tadiab","tinj", 'NOxTheoryppmM3','EngTq', 'ExhaustT', 'EnginT','SCRoutgps'])
# T1=OBD.Wheelspeed
# gt100= OBD[OBD['Wheelspeed'].gt(100)].index
# print(gt100)
OBD=pd.read_csv(path)
# T=OBD.iloc[51937:51937+60]
# T=OBD.iloc[7492:7492+60]
# T=OBD.iloc[1035:1035+60]
T=OBD
# Train=OBD.iloc[:10000]
# EnginT=T['EnginT'].to_numpy()
# ExhaustT=T['ExhaustT'].to_numpy()
intakeRPMlist=T['engrpm'].to_numpy()
intakePlist=T['IntakekPa'].to_numpy()*1000
intakeTlist=T['IntakeT'].to_numpy()+273
Fuelconskgph=T['Fuelconskgph'].to_numpy()
EGRkgph=T['EGRkgph'].to_numpy()
# +np.random.normal(0,20, T['EGRkgph'].shape)
Airinkgph=T['Airinkgph'].to_numpy()
NOxActual=T['SCRinppm'].to_numpy()
# Wheelspeed=T['Wheelspeed'].to_numpy()

# Engpwr=T['Engpwr'].to_numpy()
RailMPa=T['RailMPa'].to_numpy()
NOxActual=T['NOxActual'].to_numpy()
SCRingps=T['SCRingps'].to_numpy()

# SCRreduce=SCRingps-SCRoutgps
# xNOxActual=T['xNOxActual'].to_numpy()
Anomalyindex=T['Anomalyindex3'].to_numpy()
# Tadiab=T['TadiabModel3'].to_numpy()
# trespercent=T['tres/tcycle'].to_numpy()
# xo2intake=T['xo2intake'].to_numpy()
NOxTheoryppm=T['NOxTheoryppmM3'].to_numpy()
EngTq=T['EngTq'].to_numpy()
Accpedalpos=T['accelpedalpos'].to_numpy()
#Paramset3
Instfuelecon=T['Instfuelecon'].to_numpy()
accelpedalpos=T['accelpedalpos'].to_numpy()
ExhaustT=T['ExhaustT'].to_numpy()
Wheelspeed=T['Wheelspeed'].to_numpy()
EngTq=T['EngTq'].to_numpy()

OBD['gps_Acc']=OBD['GPSspeed'].diff()/OBD['GPSiter'].diff()
OBD['KineticIntensity']=OBD['gps_Acc']/OBD['GPSspeed']**2
OBD['EngPwr']=OBD['EngTq']*0.73756214927727/5252*OBD['engrpm']

# X1=T['X1'].to_numpy()
# X2=T['X2'].to_numpy()
# X3=T['X3'].to_numpy()
# X4=T['X4'].to_numpy()
# y=T['y'].to_numpy()


#Combustion chamber Parameters
S = 0.124#stroke (m)
B = 0.107#bore (m)
a = 0.5*S#crank radius (m) S = 2a
l = 3.5*a#connecting rod length (m)  l/a = 3-4 for small and medium size engine
Vdis = (np.pi*B**2)/4 * S;  #cylinder displacement (m^3)6 6.7L

cr = 17.3#compression ratio
Ru = 8.31434# Gas constant J/(mole K)
Vc = Vdis/(cr-1)#clearance volume (m^3)
gamma=1.35#polytropic ratio/ specific heat ratio
LHV=42.64e6# Lower Heating Value J/kg
nc=0.9; #Combustion Efficiency

IVO=-9 #Degrees ATDC

MW_fuel = 0.19065#kg/mol
MW_O2 = 0.032#kg/mol
MW_product = 0.02885#kg/mol
MW_air = 0.029#kg/mol
thetap=3 #premixed combustion duration degrees
#%%
path='C:/Users/panne027/Google Drive/NOx Model3/100000Data.csv'
OBD.to_csv(path, index=False)
#%% Calculations

Vivo= (2*a*np.cos(IVO)+np.sqrt(4*a**2*np.cos(IVO)**2-4*(a**2-l**2)))/2*np.pi*B**2/4
Tpeak=intakeTlist*(Vivo/Vc)**(gamma-1)
Ppeak= (Tpeak/intakeTlist)**(gamma/(gamma-1))*intakePlist

eqratio= 14.37/(Airinkgph/(Fuelconskgph))
phinox=0.9
#intake oxygen concentration
xfuel= Fuelconskgph/(Airinkgph+EGRkgph+Fuelconskgph)
Exhaustkgph=Airinkgph+Fuelconskgph
xexh=(0.21*Airinkgph-636.68/190.65*Fuelconskgph)/(Exhaustkgph-EGRkgph)
xexhn2=0.79*Airinkgph/(Exhaustkgph+EGRkgph)
xexhco2=13.883*0.044*Fuelconskgph/(Exhaustkgph+EGRkgph)
xexhh2o=12.026*0.018*Fuelconskgph/(Exhaustkgph+EGRkgph)
xo2 = abs((0.21*Airinkgph+xexh*EGRkgph)/(Airinkgph+EGRkgph+Fuelconskgph))
xo2intake=abs((19.90/phinox+xexh*EGRkgph/0.032/Fuelconskgph*MW_fuel)/(1+19.90/phinox*4.76+EGRkgph/Fuelconskgph*(xexh/0.032+xexhn2/0.028+xexhco2/0.044+xexhh2o/0.018)))
xn2 = 1-xo2intake-xfuel

#Injection duration equal to duration of combustion
fuelinjrate=0.86*7*np.pi*0.00018**2/4*np.sqrt(2*872*(RailMPa*10**6-intakePlist)) #7 holes 0.007" diameter
# fuelinjrate=Fuelconskgph*2/(6*intakeRPMlist*60)
tinj=Fuelconskgph/(fuelinjrate*30*intakeRPMlist)
thetainj=tinj*intakeRPMlist/60*360
Ea=618840/(50+25);
mps=2*S*intakeRPMlist/60;

# thetaID=(0.36+0.22*mps)*np.exp(Ea*(1/(Ru*intakeTlist*cr**(gamma-1))-1/17190)+(21.2/(intakePlist*cr**gamma-12.4)**0.63))
thetaID=3.45*np.exp(2100/intakeTlist)*(intakePlist)**(-1.02)/1000*intakeRPMlist/60*360
beta=0.45*thetaID/Fuelconskgph/1000000*intakeRPMlist*60*6 #On the Premixed Combustion in a Direct-Injection Diesel Engine 
# beta=0.5/thetainj
# tprem=
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

#%%
xNOxActual=SCRingps/0.040/1000*3600/Exhaustkgph*MW_product
dxNOxActual=xNOxActual[1:]-xNOxActual[:-1]
EINOxActual=SCRingps/(Fuelconskgph/3600)

#%% lead
NOxActuallead=[]
NOxActuallead[:]=NOxActual[1:]
NOxActuallead=np.append(NOxActuallead, NOxActual[len(NOxActual)-1])

EINOxActuallead=[]
EINOxActuallead[:]=EINOxActual[1:]
EINOxActuallead=np.append(EINOxActuallead, EINOxActual[len(EINOxActual)-1])

xNOxActuallead=[]
xNOxActuallead[:]=xNOxActual[1:]
xNOxActuallead=np.append(xNOxActuallead, xNOxActual[len(xNOxActual)-1])

tcycle=1/intakeRPMlist*60*2
#%% Curve fit
X1 = Tadiab
X2 = tres/intakeRPMlist
X3= abs(xo2intake)*(Airinkgph+EGRkgph)/3600*0.032/MW_product
# o2kgph= abs(xexh*(Airinkgph+Fuelconskgph)-xo2*Airinkgph)/3600/0.032

o2kgph= abs(xo2*(Airinkgph+Fuelconskgph))/3600/0.032
#%% Curve fit
X1 = Tadiab
X2 = tres/intakeRPMlist
# X3= abs(xo2intake)
X3=o2kgph
# EngTq
# X4=Accpedalpos
X=np.array([X1, X2, X3])
# y=xNOxActuallead[:]
y=NOxActual[:]

#%% Curvefit function
def xNOxpredict(X, a,b):
    # X1 =X[:,0]
    # X2= X[:,1]
    # X3=X[:,2]
    # return a+b*X1+c*X2
    # return a*(X[1]**b)*(X[2]**c)*(X[0]**(d))*np.exp(-e/X[0])
    return a*((X[1]*6)**1)*((X[2])**0.5)*np.exp(-b/X[0])
 
    # return a*(X2**1)*(X3**0.5)*(X1**(0))*np.exp(-e/X1)
# NOxTheoryppm=2.6e7*((tres/intakeRPMlist*6)**1)*(o2kgph**0.5)*(Tadiab**0)*np.exp(-1100/Tadiab)
# 
# y=dxNOxActual[:]
# p0= 200,0.5, 0.5, 0.5,3000
# p0 = 1, 0.5, 0.5,0.5,3000
# p0=1000
p0= 2.6e7, 4500
fitParams, fitCovar= curve_fit(xNOxpredict, X, y , p0, maxfev=1000000, method='trf', bounds=((0, 0),(np.inf, np.inf)))
# 
 # ,bounds=((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf),(np.inf, np.inf, np.inf,np.inf,np.inf))
# , bounds=((-np.inf, -np.inf, -np.inf, 30000),(np.inf, np.inf, np.inf, 50000))
 # , bounds=((-np.inf, -np.inf, -np.inf, 0),(np.inf, np.inf, np.inf, np.inf)
# fitParams=[18696835.9, 4705.13]
print(fitParams)
print(fitCovar)

# EINOxTheory=[]
NOxTheoryppm=xNOxpredict(X, fitParams[0], fitParams[1])

#%%
# fitParams=[7.91909801e-07, 2.98964332e-01, 6.08063090e-01, 7.95716790e-01, 2.17349162e+03]#FDT
# fitParams=[ 1.33554903e+03, -1.81638752e-01,  3.89034367e-01, -1.30603646e+00,  5.04475026e+03] #TB
# EINOxTheory= EINOxpredict((X1[:],X2[:], X3[:]),fitParams[0], fitParams[1],fitParams[2], fitParams[3])
xNOxTheory = xNOxpredict(X ,fitParams[0], fitParams[1] ,fitParams[2], fitParams[3],fitParams[4])
#, ,,fitParams[5]

NOxTheoryppm=[]
# molesNOx= EINOxTheory/38 * Fuelconskgph[:] #moles per hour
molesNOx= xNOxTheory*Exhaustkgph/MW_product
totalproductmoles= (Nn2[:] + No2[:] + 13.883 + 12.026)* Fuelconskgph[:]/MW_fuel+molesNOx #moles per hour
NOxTheoryppm= molesNOx/totalproductmoles*1e6
NOxTheoryppm= xNOxTheory*1000000

NOxTheoryppmlag=np.array([])
NOxTheoryppmlag=np.append(NOxTheoryppmlag,NOxActual[0])
NOxTheoryppmlag=np.append(NOxTheoryppmlag,NOxTheoryppm[:-1])
Error=NOxTheoryppmlag-NOxActual 
AbsError= np.abs(NOxTheoryppmlag-NOxActual)
# NOxTheoryppmlag=np.array(NOxTheoryppmlag)
# SCRingpsTheory= EINOxTheory/ 3600 * Fuelconskgph[:]

# Metrics
r_value= scipy.stats.linregress(NOxActual[:], NOxTheoryppmlag[:])
R2value=r_value[2]**2
mae=mean_absolute_error(NOxActual[:], NOxTheoryppmlag[:])
rmse=np.sqrt(mean_squared_error(NOxActual[:], NOxTheoryppmlag[:]))
print('[R2,RMSE,MAE]= [{} {} {}]'.format(R2value,rmse,mae))

#%% EINOx to NOxppm
NOxTheoryppm=[]
molesNOx=NOxTheorykgph/0.040/MW_fuel
# molesNOx= EINOxTheory/0.040 * Fuelconskgph[:] #moles per hour
# molesNOx= xNOxTheory*Exhaustkgph/MW_product
# totalproductmoles= (xexhn2[:]*Exhaustkgph/0.028 + xexh*Exhaustkgph/0.032 + 13.883 + 12.026)* Fuelconskgph[:]/MW_fuel + molesNOx #moles per hour

totalproductmoles= (Nn2 + No2 + 13.883 + 12.026)* Fuelconskgph[:]/MW_fuel + molesNOx #moles per hour
NOxTheoryppm= molesNOx/totalproductmoles*1e6 
#%%
NOxTheoryppm=2.6e7*((tres/intakeRPMlist*6)**1)*(o2kgph**0.5)*(Tadiab**0)*np.exp(-1100/Tpeak)
#%%
NOxTheoryppmlag=np.array([])
NOxTheoryppmlag=np.append(NOxTheoryppmlag,NOxActual[0])
NOxTheoryppmlag=np.append(NOxTheoryppmlag,NOxTheoryppm[:-1])
Error=NOxTheoryppmlag-NOxActual 
AbsError= np.abs(NOxTheoryppmlag-NOxActual)


# NOxTheoryppmlag=np.array(NOxTheoryppmlag)
# SCRingpsTheory= EINOxTheory/ 3600 * Fuelconskgph[:]

# Metrics
r_value= scipy.stats.linregress(NOxActual[:], NOxTheoryppmlag[:])
R2value=r_value[2]**2
mae=mean_absolute_error(NOxActual[:], NOxTheoryppmlag[:])
rmse=np.sqrt(mean_squared_error(NOxActual[:], NOxTheoryppmlag[:]))
print('[R2,RMSE,MAE]= [{} {} {}]'.format(R2value,rmse,mae))

# Baseline plot
plt.rcParams.update({'font.size': 60})
plt.figure(figsize=(35,35))
plt.scatter(NOxActual[:], NOxTheoryppmlag[:], c=abs(NOxTheoryppmlag[:]-NOxActual[:]), cmap='viridis' )
plt.plot(range(0,int(1000), 1),range(0,int(1000), 1))
plt.xlim(0, 800)
plt.ylim(0, 800)

# NOxTheoryppm=2.6e7*((tres/intakeRPMlist*6)**1)*(xo2**0.5)*(Tadiab**0)*np.exp(-1100/Tadiab)
plt.title('Regressed Baseline Model Prediction',fontsize=60)
# plt.title('Training data Neural Network',fontsize=45)
plt.xlabel('NOx Observed /ppm', fontsize=60)
plt.ylabel('NOx Predicted /ppm', fontsize=60)
cbar=plt.colorbar()
plt.clim(0,1000)
cbar.set_label('Divergence /ppm', fontsize=60)

#%%
def EINOxpredict2(X):
    X1, X2,X3=X
    # return a+b*X1+c*X2
    return 2.6e7*((X2/1000)**1)*(X3**0.5)*np.exp(-1100/X1)
NOxTheorykgph=[]
EINOxTheory=[]
NOxTheorykgph= EINOxpredict2((X1[:],X2[:], X3[:]))

#%% split
x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.5)

#%% linear regression
reg1=LinearRegression().fit(x_train,y_train)
print(reg1.intercept_, reg1.coef_)
EINOxTheory=[]
EINOxTheory=reg1.predict(X)

#%% ridge
reg2=Ridge(alpha=1,max_iter=1000000).fit(x_train,y_train)
print(reg2.intercept_, reg2.coef_)
EINOxTheory=[]
EINOxTheory=reg2.predict(X)
#%% Bayesianridge
reg3=BayesianRidge(n_iter=1000000).fit(x_train,y_train)
print(reg3.intercept_, reg3.coef_)
EINOxTheory=[]
EINOxTheory=reg3.predict(X)

#%% Kinetic Intensity
acc=0
v2aero=0
for i in range(1,len(OBD['GPSspeed'])):
    if OBD['pass'][i]==OBD['pass'][i-1]:
        acc+=abs(0.5*((OBD['GPSspeed'][i]/3.6)**2-(OBD['GPSspeed'][i-1]/3.6)**2)+9.81*(OBD['Totdist'][i]-OBD['Totdist'][i-1])*1000)
        v2aero+=(OBD['GPSspeed'][i]/3.6)**3*(OBD['GPSiter'][i]-OBD['GPSiter'][i-1])
ki=acc/v2aero
print('Kinetic Intensity= {}'.format(ki))
# %% neural network data
X1nn=Tadiab
X2nn=tres
X3nn=xo2
X4nn=EngTq

Xnn=np.array([X1nn, X2nn, X3nn, X4nn]).transpose()

x_trainnn, x_testnn, y_trainnn, y_testnn=train_test_split(Xnn,y,test_size=0.5)

#%%neural network

# reg4 = MLPRegressor(, max_iter=1000000).fit(x_train, y_train)
reg4 = MLPRegressor(hidden_layer_sizes=(100,),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
               learning_rate='constant', learning_rate_init= 0.01, power_t=0.5, max_iter=1000000, shuffle=True,
               random_state=10, tol=0.0001, verbose=False, warm_start=False,
                early_stopping=False, validation_fraction=0.1, beta_1=0.95, beta_2=0.999,momentum=0.9,
               epsilon=1e-08).fit(X,y)

# print(reg4.intercepts_, reg4.coefs_)

EINOxTheory=[]
EINOxTheory=reg4.predict(X)

# momentum=0.9,nesterovs_momentum=True,

#%% Tensorflow

# def build_model():
#     reg5 = keras.Sequential([
#         layers.Dense(64, activation = 'relu', input_shape=[len(NOxActual)]),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1)
#     ])
    
#     optimizer=tf.keras.optimizers.RMSprop(0.001)
#     reg5.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])
    
#     return reg5

# def norm(x):
#   return (x - train_stats['mean']) / train_stats['std']
# normed_train_data = norm(x_train)
# normed_test_data = norm(x_test)

# EINOxTheory=[] 
# EPOCHS=1000

# EINOxTheory = reg5.fit(
#   normed_train_data, train_labels,
#   epochs=EPOCHS, validation_split = 0.2, verbose=0,
#   callbacks=[tfdocs.modeling.EpochDots()])

input_layer = Input(shape=(X.shape[1],))
dense_layer_1 = Dense(20, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(5, activation='relu')(dense_layer_2)
output = Dense(1)(dense_layer_3)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
history = model.fit(x_train, y_train, batch_size=2, epochs=30, verbose=1, validation_split=0.2)

EINOxTheory=[]
EINOxTheory=model.predict(X)
# score = model.evaluate(x_test, y_test, verbose=1)


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
# ax.set(xlabel='NOx Observed /ppm', ylabel='NOx Predicted /ppm')
h.set_axis_labels('NOx Observed /ppm', 'NOx Predicted /ppm')
h.fig.suptitle("Jointplot for NOxModel3 Prediction")
# sns.lineplot(range(0,int(1000), 1),range(0,int(1000), 1))
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
# index=np.arange(51937,51937+60)
index=np.arange(6000,6000+60)
# index=np.arange(70000,70000+60)

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

ax1[0].scatter(NOxActual[clusterindex], NOxTheoryppm[clusterindex] )
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
ax1[1].set_ylabel('Tadiab /K')
ax1[1].set_xlabel('Timestamp')
ax1[1].grid()
ax1[1].minorticks_on()
ax1[2].scatter(clusterindex,Wheelspeed[clusterindex])
ax1[2].set_ylabel('Wheelspeed')
ax1[2].grid()
ax1[2].minorticks_on()
ax1[2].set_xlabel('Timestamp')

ax1[3].scatter(clusterindex,EngTq[clusterindex])
ax1[3].set_ylabel('Engine Load')
ax1[3].grid()
ax1[3].minorticks_on()
ax1[3].set_xlabel('Timestamp')


ax1[4].scatter(clusterindex, intakeRPMlist[clusterindex])
ax1[4].set_ylabel('Engine RPM')
ax1[4].grid()
ax1[4].minorticks_on()
ax1[4].set_xlabel('Timestamp')

ax1[5].scatter(clusterindex, ExhaustT[clusterindex])
ax1[5].set_ylabel('ExhaustT')
ax1[5].grid()
ax1[5].minorticks_on()
ax1[5].set_xlabel('Timestamp')

ax1[6].scatter(clusterindex, SCRreduce[clusterindex])
ax1[6].set_ylabel('SCRingps-SCRoutgps')
ax1[6].grid()
ax1[6].minorticks_on()
ax1[6].set_xlabel('Timestamp')

#%%
time=np.arange(1,61)
plt.rcParams.update({'font.size': 18})
plt.style.use('seaborn-ticks')
fig1,ax1=plt.subplots(4, sharex=True, sharey=False, figsize=(20,25))

ax1[0].set_title('Data Visualization')
ax1[0].plot(time, y[index])
ax1[0].set_ylabel('$x_{NO_{x}}$ Observed')
ax1[0].grid()
ax1[0].minorticks_on()

ax1[1].plot(time, X3[index])
ax1[1].set_ylabel('$X_{O_{2}}$')
ax1[1].grid()
ax1[1].minorticks_on()

ax1[2].plot(time,X2[index])
ax1[2].set_ylabel('Dimensionless $t_{comb}$')
ax1[2].grid()
ax1[2].minorticks_on()

ax1[3].plot(time,X1[index])
ax1[3].set_ylabel('$T_{adiab}$ /K')
ax1[3].grid()
ax1[3].minorticks_on()


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
