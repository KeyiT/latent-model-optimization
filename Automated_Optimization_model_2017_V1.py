# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 17:02:36 2017

@author: Minglei
"""
import hp816x_instr
import Ke26XXA
from Ke26XXA import Ke26XXA
import HP11896A
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

#%%
k1 = Ke26XXA() #Keithley instrument
k1.connect('GPIB0::28::INSTR') #connects to Keithley
Init_h12 = [0.5, 0.5]
Init_theta12 = [0.5, 0.5]
Imax = 36e-3
#%%
#connect the polarization controller 11896A
Input1 = HP11896A.HP11896A()
Input1.connect('GPIB0::21::INSTR')
model = customized.PhysicalModel(Init_h12, Init_theta12, 1, 0, k1, 'b', 'a', Imax)
model.hpmainframe_connect('GPIB0::26::INSTR')
model.hpmainframe_laser('on')


#%% set the Initial Pos.state
#Input1.SetPaddlePos('1', (518))
#Input1.SetPaddlePos('2', (435))  
#Input1.SetPaddlePos('3', (486))
#Input1.SetPaddlePos('4', (173))

#%%
Timematrix = []         # measurement timetable
OutputpowerTop = []
OutputpowerBottom = []
Currents = []
T0 = time.time()
Timematrix.append(T0)   # set the initital time

for iteration in xrange(0,4):
    
    Timematrix.append(time.time())
    Input1.SetPaddlePos('1', (random.randint(0,999)))
    Input1.SetPaddlePos('2', (random.randint(0,999)))  
    Input1.SetPaddlePos('3', (random.randint(0,999)))
    Input1.SetPaddlePos('4', (random.randint(0,999)))
    time.sleep(1.0e-2)
    Timematrix.append(time.time())
    model.setPWMUnit('W')
    model.setAvgtime(10e-3)
    model.on()
    model.train_and_optimize(sample_numbers=[5,5])
    Timematrix.append(time.time())
    OutputpowerBottom.append(model.Output_PWM(0))
    OutputpowerTop.append(model.Output_PWM(1))
    Currents.append()
    time.sleep(5.0)
    iteration = iteration + 1
    
#%%
model.setPWMUnit('dBm')
model.off()
model.hpmainframe_laser('off')

Timematrix[:] = [x-T0-1 for x in Timematrix]   #get the recording time from 0
Timematrix = np.asarray(Timematrix)
                            
#%% save iv data
baseFolder = 'D:/Minglei/ActiveIMEchip_2017test/SiEPICPassive193nm_chip7_PR_noendlessL100um/'
matfile_name='OptOuput_MHSweepMax30mA_TM0dBminput_1550_2nm.mat'    
matDict = dict()
matDict['data'] = dict()
matDict['data']['voltage'] = voltages_sweep
matDict['data']['current'] = currents_sweep
matDict['data']['PoutTop_2ndMH'] = OutputpowerTop_sweepP1
matDict['data']['PoutBottom_2ndMH'] = OutputpowerBottom_sweepP1
matDict['data']['PoutTop_1stMH'] = OutputpowerTop_sweepP2
matDict['data']['PoutBottom_1stMH'] = OutputpowerBottom_sweepP2
savemat(baseFolder+matfile_name, matDict)
