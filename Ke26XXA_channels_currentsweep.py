# -*- coding: utf-8 -*-
"""
Created on Thr May 18 12:00:08 2017

@author: Minglei
"""
import numpy as np;
from scipy.interpolate import interp1d
from scipy.io import loadmat

class Ke26XXA_channels_currentsweep(object):
    name = 'Keithley 26xxA channels' 
       
    def __init__(self, k, channel, imin, imax, NPLC = 1.0, current_limit=28.0e-3, name = 'PN'):
        self.keithley = k;
        self.ch = channel;
        self.imin = imin;
        self.imax = imax;
        self.current_limit = current_limit;        
        self.keithley.setCurrentMeasurementRange(self.ch, self.current_limit)
        self.keithley.setAutoZeroMode(self.ch, 'auto')
        self.keithley.setNPLC(self.ch, NPLC);
        self.name = name;
        self.resdatafile = 'D:/Minglei/ActiveIMEchip_2017test/IME_chip21_PR_noendless/PRAC_endless.mat'

    def on(self):
        self.keithley.outputEnable(self.ch, True)
        
    def off(self):
        self.keithley.outputEnable(self.ch, False)
  
    # Sets the voltage for a channel. If cvMode is true sets channel automatically to constant voltage mode
    def setVoltage(self, voltage):
        self.keithley.setVoltage(self.ch, voltage)
#        if(voltage > self.vmax + 0.1): 
#            print 'voltage larger than maximum channel voltage'
#            self.keithley.setVoltage(self.ch, self.vmax)
#        elif( voltage < 0.001):
#            print 'voltage too small'
#            self.keithley.setVoltage(self.ch, self.vmin)                
#        else:
#            self.keithley.setVoltage(self.ch, voltage)
            
    def setCurrent(self, current):
        self.keithley.setCurrent(self.ch, current)
        if(current > self.imax + 2e-3): 
            print 'current larger than maximum channel current'
            self.keithley.setCurrent(self.ch, self.imax)
        elif(current < 1e-4):
            print 'current too small'
            self.keithley.setCurrent(self.ch, self.imin)
            
    def setPower(self, power, vstep=0.01, power_accuracy = 0.1e-3):
        voltage = self.getVoltage()
        current = self.getCurrent()
        curr_power = voltage*current;
        
        if(curr_power < power):
            low_side = True
        else:
            low_side = False
            vstep = vstep*(-1.0)
        
        while(abs(power-curr_power) > power_accuracy):
            voltage = voltage + vstep
            self.setVoltage(voltage)
            
            voltage = self.getVoltage()
            current = self.getCurrent()
            curr_power = voltage*current;
            
            print str(curr_power*1e3) + 'mW     ' +  str(vstep)
            
            if(curr_power > power and low_side):
                vstep = vstep*(-1.0)
                vstep = vstep/2;
                low_side = False
                
            if(curr_power < power and low_side == False):
                vstep = vstep*(-1.0)
                vstep = vstep/2;
                low_side = True
                
            if(abs(vstep) < 1.0e-4):
                print 'Minimum voltage step reached without meeting power accuracy'
                break;

    def getVoltage(self):
        return self.keithley.getVoltage(self.ch)
        
    def getCurrent(self):
        return self.keithley.getCurrent(self.ch)
    
    #call to self populate IV curves
    def getIV(self, numPts=100):
        #numPts = 101;\
        currents = np.linspace(self.imin, self.imax, numPts)
        voltages = np.zeros(np.shape(currents))
        
#        v0 = self.getVoltage();
        self.setCurrent(self.imin)
        self.on()
        for ii, II in enumerate(currents):
            self.setCurrent(II)
            voltages[ii] = self.getVoltage()
            currents[ii] = self.getCurrent()
        
        self.setCurrent(self.imin)
        
#        self.IVfunc = interp1d(voltages, currents.T, kind='cubic')
#        power = voltages*currents;
#        self.VPfunc_private = interp1d(power, voltages.T, kind = 'cubic', fill_value=self.vmin)
        
#        self.pmin = power.min()
#        self.pmax = power.max()
        self.off()
        return (voltages, currents)
   
    def getPower(self):
        voltage = self.getVoltage()
        current = self.getCurrent() 
        power = voltage*current
        return (power)           
        
    #to avoid out of range values
    def VPfunc(self, powervals):
        powervals = np.array(powervals)
        powervals[powervals < self.pmin] = self.pmin;
        powervals[powervals > self.pmax] = self.pmax;
        
        return self.VPfunc_private(powervals)
        
    def loadresdata(self):
        resdata = loadmat(self.resdatafile, squeeze_me=True, struct_as_record=False);
        return resdata     
            
        