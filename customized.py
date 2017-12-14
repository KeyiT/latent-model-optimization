from minglei_complete_model import MingLeiModel
import hp816x_instr
import time
import Ke26XXA
reload(Ke26XXA)
from Ke26XXA import Ke26XXA        
import math
import numpy as np
#from sympy.slovers import solve
#from sympy import Symbol

class PhysicalModel(MingLeiModel):
    def __init__(self, init_hidden_vars, init_params, opt_slot, opt_chn, keith_dev, keith_chn1, keith_chn2, keith_imax):
        self.opt_slot = opt_slot
        self.opt_chn = opt_chn
        self.keith_dev = keith_dev
        self.keith_chn1 = keith_chn1
        self.keith_chn2 = keith_chn2       
        self.keith_imax = keith_imax 
        self.hp_mainframe = hp816x_instr.hp816x() 
        super(PhysicalModel, self).__init__(init_hidden_vars, init_params)
        print('ok1')

    def observe(self, params):
        # TODO: return p3 from your machine (in power unit)
        self.set_params(params)
        time.sleep(0.001)  ##unit: s
        p3 = self.hp_mainframe.readPWM(self.opt_slot, self.opt_chn)
        p3_dBm = 10*np.log10(p3)+30
        print(p3_dBm)
        return p3

    def hpmainframe_connect(self, visaAddr):
        self.hp_mainframe.connect(visaAddr, reset=0, forceTrans=1) #connects to the laser
  
    def setAvgtime(self, Avgtime):
        self.hp_mainframe.setPWMAveragingTime(1, 1, Avgtime)   #unit: s
        self.hp_mainframe.setPWMAveragingTime(1, 0, Avgtime)
        
    def setPWMUnit(self, Unit):
        self.hp_mainframe.setPWMPowerUnit(self.opt_slot, 0, Unit)
        self.hp_mainframe.setPWMPowerUnit(self.opt_slot, 1, Unit)
 
    def on(self):
        self.keith_dev.outputEnable(self.keith_chn1, True)
        self.keith_dev.outputEnable(self.keith_chn2, True)
        
    def off(self):
        self.keith_dev.outputEnable(self.keith_chn1, False)
        self.keith_dev.outputEnable(self.keith_chn2, False)
    
    def set_params(self, params):
        self.params = params
        alpha = 97.08
        beta = 0.02539
        R = 200
        print(params)
#        a = []
#        b = []
#        eqn1 = a[0]*current1**4 + a[1]*current1**3 + a[2]*current1**2 - (params[0] - beta)/alpha
#        eqn2 = b[0]*current2**4 + b[1]*current1**3 + b[2]*current1**2 - (params[1] - beta)/alpha
#        slove(eqn1, current1, )

        coeff = [
            3.5833e+05, 31480, 157.6143, 159.2961, 0, -(params[0] - beta) / alpha,
            3.5833e+05, 31480, 157.6143, 159.2961, 0, -(params[1] - beta) / alpha,
        ]
        roots = [np.roots(coeff[0]), np.roots(coeff[1])]

        for i in range(0, len(roots)):
            roots_ = roots[i]
            roots_ = list(filter(
                lambda r_: np.abs(np.imag(r_)) < 1E-8,
                roots_
            ))
            roots[i] = list(filter(
                lambda r_: r_ >= 0 & r_ < 0.036,
                roots_
            ))

            if len(roots[i]) == 0:
                raise ValueError("invalid roots")

        print("\ncurrent: ")
        print(roots)

        """
        # TODO: set theta1 and theta2 to your machine. params=[theta1, theta2]
        if params[0] < beta and params[1] < beta:
            current1 = math.sqrt((params[0]-beta+2*math.pi)/(alpha*R))
            current2 = math.sqrt((params[1]-beta+2*math.pi)/(alpha*R))
        if params[0] < beta and params[1] > beta:
            current1 = math.sqrt((params[0]-beta+2*math.pi)/(alpha*R))
            current2 = math.sqrt((params[1]-beta)/(alpha*R))
        if params[0] > beta and params[1] < beta:
            current1 = math.sqrt((params[0]-beta)/(alpha*R))
            current2 = math.sqrt((params[1]-beta+2*math.pi)/(alpha*R))
        if params[0] > beta and params[1] > beta:
            current1 = math.sqrt((params[0]-beta)/(alpha*R))
            current2 = math.sqrt((params[1]-beta)/(alpha*R))         
        if current1 > self.keith_imax + 1e-3:
            print('current larger than maximum channel1 current')
            current1 = math.sqrt((params[0]-beta-2*math.pi)/(alpha*R))
        if current2 > self.keith_imax + 1e-3:
            print('current larger than maximum channel2 current')
            current2 = math.sqrt((params[1]-beta-2*math.pi)/(alpha*R))
        """

        self.keith_dev.setCurrent(self.keith_chn1, roots[0][0])
        self.keith_dev.setCurrent(self.keith_chn2, roots[1][0])


# instruction:
# 1. implement observe and set_params in PhysicalModel
# 2. initialize: model = PhysicalModel([0.5, 0.5], [0.5, 0.5])
# 3. set slot and channel: model.set_slot_chn(1, 1)
# 4. find h1, h2, theta1, theta2: model.train_and_optimize()
# 5. get it run along with your machine. you can keep calling
        # model.train_and_optimize() to calibrate your machine in each time step
