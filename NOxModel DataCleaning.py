# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:16:22 2020

@author: panne027
"""

#%% Read data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, leastsq
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns

# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Activation, Dropout
# import seaborn as sns

path='C:/Users/panne027/Google Drive/NOx Model3/160000DataFoodDelivery.csv'
# path='D:/Google Drive/NOx Model3/200000DataRefuse.csv'
print(path)
# OBD1=pd.read_csv('D:/Google Drive/NOx Model3/500000DataDrayage.csv', usecols=['time',"EngIntakeManifold1Temp","EngTurboBoostPress","EngSpeed","EngFuelRate","EngExhstGsRcrcltionMassFlowRate",
# OBD1=pd.read_csv(path, usecols=['time',"EngIntakeManifold1Temp","EngTurbo1CompressorIntakePress","TransClutch_ConverterInputSpeed","EngFuelRate","EngExhstGsRcrcltionMassFlowRate",
#                                                                                            "EngInletAirMassFlowRate", 'EngInjectorMeteringRail1Press','Aftertreatment1IntakeNOx', 'WheelBasedVehicleSpeed'])
#not yard tractor:
OBD1=pd.read_csv(path, usecols=['time','Date',"EngIntakeManifold1Temp","EngTurboBoostPress","EngSpeed","EngFuelRate","EngExhstGsRcrcltionMassFlowRate","EngInletAirMassFlowRate", 'EngFuelDeliveryPress','Aftertreatment1IntakeNOx', 'WheelBasedVehicleSpeed','EngInstantaneousFuelEconomy', 'ActualEngPercentTorque', 'Aftertreatment1ExhaustGasTemp1','AccelPedalPos1'])
# 200000DataYardTractor
# 500000DataDrayage
# 160000DataFoodDelivery
# 200000DataRefuse
OBD=OBD1[OBD1['Aftertreatment1IntakeNOx']>0]
OBD=OBD[OBD['Aftertreatment1IntakeNOx']<2500]
OBD.replace([np.inf, -np.inf], np.nan).dropna()
OBD=OBD[OBD['EngSpeed']!=0]
OBD=OBD[OBD['EngFuelRate']!=0]
OBD=OBD[OBD['EngIntakeManifold1Temp']!=0]


# EngTurbo1CompressorIntakePress
# OBD=OBD1[OBD1['EngExhstGsRcrcltionMassFlowRate']!=0]

#EngTurboBoostPress
# T1=OBD.Wheelspeed
# gt100= OBD[OBD['Wheelspeed'].gt(100)].index
# print(gt100)

# T=OBD.iloc[51937:51937+60]
# T=OBD.iloc[7492:7492+60]
# T=OBD.iloc[1035:1035+60]
#%%
Month=OBD['Date'].str.split(pat="/", expand=True)[0]
Day=OBD['Date'].str.split(pat="/", expand=True)[1]
Year=OBD['Date'].str.split(pat="/", expand=True)[2]






#%%
