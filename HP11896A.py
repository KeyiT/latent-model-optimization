"""
Created on Thu Aug 24 10:16:47 2017
Class function for HP 11896A
@author: Minglei
"""
import visa
from numpy import *
import numpy 
    
class HP11896A(object):
    name = "Polarization controller 11896A"

#    def __init__(self):
    
    def connect(self, visaAddr):
        self.rm = visa.ResourceManager()
        self.instr = self.rm.open_resource(visaAddr)                     
#        print(self.instr.ask('*IDN?\n')) #Requests the device for identification
        self.instr.clear()
        
    def WaitCommand(self):
        self.instr.write('*WAI')
        
    def ClearStatus(self):
        self.instr.write('*CLS')
        
    def ResetInstr(self):
        self.instr.write('*RST')
        
    def SetPaddlePos(self, Chn, Pos):
        self.instr.write(':PADD'+Chn+':POS '+str(Pos))
        
    def StartScan(self):
        self.instr.write(':INITiate:IMMediate')
        
    def StopScan(self):
        self.instr.write(':ABORt')
   
    def SetScanRate(self, Rate):
        self.instr.write(':SCAN:RATE '+str(Rate))   ## scan rate from 1 to 8, or MAX and MIN
        
    def GetScanRate(self):
        rate = str(self.instr.query(':SCAN:RATE?')).rstrip()
        return rate
    
    def GetScanTimer(self):
        timer = str(self.instr.query(':SCAN:TIMer?')).rstrip()
        return timer
    
    def ClearTimer(self):
        self.instr.write(':SCAN:TIMer:CLEar')
        
        