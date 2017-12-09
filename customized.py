from minglei_complete_model import MingLeiModel
import hp816x_instr
import time
import Ke26XXA
reload(Ke26XXA)
from Ke26XXA import Ke26XXA        
import math

class PhysicalModel(MingLeiModel):
    def __init__(self, init_hidden_vars, init_params, opt_slot, opt_chn, keith_dev, keith_chn1, keith_chn2, keith_imax):
        super(PhysicalModel, self).__init__(init_hidden_vars, init_params)
        self.opt_slot = opt_slot
        self.opt_chn = opt_chn
        self.keith_dev = keith_dev
        self.keith_chn1 = keith_chn1
        self.keith_chn2 = keith_chn2       
        self.keith_imax = keith_imax         

    def observe(self, params):
        # TODO: return p3 from your machine (in power unit)
        self.set_params(params)
        time.sleep(0)
        return hp816x_instr.hp816x().readPWM(self.opt_slot, self.opt_chn)
 
    def on(self):
        self.keith_dev.outputEnable(self.keith_ch1, True)
        self.keith_dev.outputEnable(self.keith_ch2, True)
        
    def off(self):
        self.keith_dev.outputEnable(self.keith_ch1, False)
        self.keith_dev.outputEnable(self.keith_ch2, False)    
    
    def set_params(self, params):
        self.params = params
        alpha = 100.8
        R = 300
        beta = 2.962
        # TODO: set theta1 and theta2 to your machine. params=[theta1, theta2]
        current1 = math.sqrt((params[0])/(alpha*R))
        current2 = math.sqrt((params[1])/(alpha*R))        
        if current1 > self.keith_imax + 1e-3:
            print('current larger than maximum channel1 current')
            current1 = math.sqrt((params[0]-beta-2*math.pi)/(alpha*R))
        if current2 > self.keith_imax + 1e-3:
            print('current larger than maximum channel2 current')
            current2 = math.sqrt((params[1]-beta-2*math.pi)/(alpha*R))
        self.keith_dev.setCurrent(self.keith_chn1, current1)
        self.keith_dev.setCurrent(self.keith_chn2, current2)


# instruction:
# 1. implement observe and set_params in PhysicalModel
# 2. initialize: model = PhysicalModel([0.5, 0.5], [0.5, 0.5])
# 3. set slot and channel: model.set_slot_chn(1, 1)
# 4. find h1, h2, theta1, theta2: model.train_and_optimize()
# 5. get it run along with your machine. you can keep calling
        # model.train_and_optimize() to calibrate your machine in each time step
