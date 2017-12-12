# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 17:02:36 2017

@author: Minglei
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.io import savemat
import HP11896A
reload (HP11896A)
import random
from random import randint
import customized
reload(customized)
from customized import PhysicalModel
import Ke26XXA
from Ke26XXA import Ke26XXA
import hp816x_instr
import HP11896A

#%%
k1 = Ke26XXA() #Keithley instrument
k1.connect('GPIB0::28::INSTR') #connects to Keithley
HPMainFrame = hp816x_instr.hp816x() #sets up 'laser' as the hp816x_instr object
HPMainFrame.connect('GPIB0::26::INSTR', reset=0, forceTrans=1) #connects to the laser
HPMainFrame.setPWMAveragingTime(1, 1, 10e-3)   #unit: s
HPMainFrame.setPWMAveragingTime(1, 0, 10e-3)
Init_h12 = [0.5, 0.5]
Init_theta12 = [0.5, 0.5]
Imax = 36e-3
#%%
#connect the polarization controller 11896A
Input1 = HP11896A.HP11896A()
Input1.connect('GPIB0::21::INSTR')
Input1.SetPaddlePos('1', (random.randint(0,999)))
Input1.SetPaddlePos('2', (random.randint(0,999)))  
Input1.SetPaddlePos('3', (random.randint(0,999)))
Input1.SetPaddlePos('4', (random.randint(0,999)))
#%%
model = customized.PhysicalModel(Init_h12, Init_theta12, 1, 1, k1, 'b', 'a', Imax)
model.on()
model.train_and_optimize()
model.off()
                            

