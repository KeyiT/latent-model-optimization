from minglei_complete_model import MingLeiModel
import hp816x_instr


class PhysicalModel(MingLeiModel):
    def __init__(self, init_hidden_vars, init_params):
        super(PhysicalModel, self).__init__(init_hidden_vars, init_params)
        self.slot = None
        self.chn = None
        
        self.keithley_chn = None

    def observe(self, params):
        # TODO: return p3 from your machine (in power unit)
        self.set_params(params)
        return hp816x_instr.hp816x().readPWM(self.slot, self.chn)
    
    def set_params(self, params):
        self.params = params
        # TODO: set theta1 and theta2 to your machine. params=[theta1, theta2]
        self.keithley.setCurrent(self.keithley_chn, current)

    def set_slot_chn(self, slot, chn):
        self.slot = slot
        self.chn = chn


# instruction:
# 1. implement observe and set_params in PhysicalModel
# 2. initialize: model = PhysicalModel([0.5, 0.5], [0.5, 0.5])
# 3. set slot and channel: model.set_slot_chn(1, 1)
# 4. find h1, h2, theta1, theta2: model.train_and_optimize()
# 5. get it run along with your machine. you can keep calling
        # model.train_and_optimize() to calibrate your machine in each time step
