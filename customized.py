from minglei_complete_model import MingLeiModel
import hp816x_instr


class PhysicalModel(MingLeiModel):
    def __init__(self, init_hidden_vars, init_params):
        super(PhysicalModel, self).__init__(init_hidden_vars, init_params)

    def observe(self, [slot, chn]):
        pass
        # TODO: return p3 from your machine (in power unit)
        return hp816x_instr_.hp816x().readPWM(slot, chn)
    
    def set_params(self, params):
        self.params = params
        # TODO: set theta1 and theta2 to your machine. params=[theta1, theta2]



# instruction:
# 1. implement observe and set_params in PhysicalModel
# 2. initialize: model = PhysicalModel([0.5, 0.5], [0.5, 0.5])
# 3. find h1, h2, theta1, theta2: model.train_and_optimize()
# 4. get it run along with your machine. you can keep calling
        # model.train_and_optimize() to calibrate your machine in each time step
