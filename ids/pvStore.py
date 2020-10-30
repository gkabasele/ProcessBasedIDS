import yaml
import math
import pdb
import numpy as np
import utils
from timeSeriesAnalysis import Digitizer, polynomial_fitting

VAR_PERIOD = 1

class PVStore(object):

    def __init__(self, descFile, data=None, slen=utils.DAY_IN_SEC):
        # name -> Process Variable
        self.vars = {}
        self.setup(descFile)
        if data is not None:
            self.detect_periodic_shape(data)

    def setup(self, descFile):
        with open(descFile) as fh:
            content = fh.read()
            desc = yaml.load(content, Loader=yaml.Loader)
            for var_desc in desc["variables"]:
                var = var_desc["variable"]
                if var["type"] == utils.DIS_COIL or var["type"] == utils.DIS_INP:
                    limit_values = [0, 1, 2]
                    pv = utils.ProcessSWaTVar(var["name"], var["type"],
                                              limit_values=limit_values,
                                              min_val=0,
                                              max_val=2,
                                              ignore=var["ignore"])
                else:
                    if 'critical' in var and "digitizer" in var:
                        limit_values = var['critical']
                        digitizer = Digitizer()
                        digitizer.deserialize(var["digitizer"])
                        ignore = var["ignore"] 
                    else:
                        limit_values = None
                        digitizer = None
                        ignore = var["ignore"]
                    pv = utils.ProcessSWaTVar(var['name'], var['type'],
                                              limit_values=limit_values,
                                              digitizer=digitizer,
                                              min_val=var['min'],
                                              max_val=var['max'],
                                              ignore=ignore)
                self.vars[pv.name] = pv

    def detect_periodic_shape(self, data):
        cont_vars = self.continous_vars()
        cont_vars_digit = {x: Digitizer(self.vars[x].min_val, self.vars[x].max_val) for x in cont_vars}

        for state in data:
            for k, v in self.vars.items():
                if not v.is_bool_var():
                    value = state[k]
                    d = cont_vars_digit[k]
                    d.online_digitize(value)

        for k, d in cont_vars_digit.items():
            x_axis = np.arange(len(d.res))
            model = polynomial_fitting(x_axis, d.res)
            self.vars[k].is_periodic = np.var(model) <= VAR_PERIOD


    def _compute_periodic_vars(self, partition, cont_vars_part):
        for state in partition:
            for k in cont_vars_part:
                cont_vars_part[k].append(state[k])

    def continous_vars(self):
        return [x for x, j in self.items() if j.kind in [utils.HOL_REG, utils.INP_REG]]

    def discrete_vars(self):
        return [x for x, j in self.items() if j.kind in [utils.DIS_COIL, utils.DIS_INP]]

    def monitor_vars(self):
        return [x for x, j in self.items() if not j.kind]

    def continuous_monitor_vars(self):
        return [x for x, j in self.items() if (j.kind in [utils.HOL_REG, utils.INP_REG] and not j.ignore)]

    def discrete_monitor_vars(self):
        return [x for x, j in self.items() if (j.kind in [utils.DIS_COIL, utils.DIS_INP] and not j.ignore)]


    def periodic_vars(self):
        return [x for x, j in self.items() if j.is_periodic]

    def __getitem__(self, key):
        return self.vars[key]

    def __setitem__(self, key, value):
        self.vars[key] = value

    def __delitem__(self, key):
        self.vars.__delitem__(key)

    def __iter__(self):
        return self.vars.__iter__()

    def values(self):
        return self.vars.values()

    def keys(self):
        return self.vars.keys()

    def items(self):
        return self.vars.items()

if __name__ == "__main__":

    #data = utils.read_state_file("../SWaT_scripts/process_variables/SWat_Dataset_normal_v1.bin")[utils.COOL_TIME:]
    data = utils.read_state_file("../SWaT_scripts/process_variables/SWat_Dataset_normal_v1_5d.bin")
    store = PVStore("./eval_swat_process/swat_process_conf_limit_hist.yml", data)
    pdb.set_trace()
