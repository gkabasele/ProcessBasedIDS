import yaml
import math
import pdb
import numpy as np
import utils

class PVStore(object):

    def __init__(self, descFile, data=None, slen=utils.DAY_IN_SEC):
        # name -> Process Variable
        self.vars = {}
        self.setup(descFile)
        if data is not None:
            self.compute_periodic_vars(data, slen)

    def setup(self, descFile):
        with open(descFile) as fh:
            content = fh.read()
            desc = yaml.load(content, Loader=yaml.Loader)
            for var_desc in desc['variables']:
                var = var_desc['variable']
                if var['type'] == utils.DIS_COIL or var['type'] == utils.DIS_INP:
                    limit_values = [0, 1, 2]
                    pv = utils.ProcessSWaTVar(var['name'], var['type'],
                                              limit_values=limit_values,
                                              min_val=0,
                                              max_val=2)
                else:
                    if 'critical' in var:
                        limit_values = var['critical']
                        if len(limit_values) == 0:
                            limit_values.extend([var['min'], var['max']])
                        limit_values.sort()
                    else:
                        limit_values = [var['min'], var['max']]
                        limit_values.sort()
                    pv = utils.ProcessSWaTVar(var['name'], var['type'],
                                              limit_values=limit_values,
                                              min_val=var['min'],
                                              max_val=var['max'])
                self.vars[pv.name] = pv

    def _compute_periodic_vars(self, partition, cont_vars_part):
        for state in partition:
            for k in cont_vars_part:
                cont_vars_part[k].append(state[k])

    def compute_periodic_vars(self, data, slen, thresh=utils.DIST):
        cont_vars = self.continous_vars()
        cont_vars_part = {x: [] for x in cont_vars}
        vars_means = {x:[] for x in cont_vars}
        nbr_partition = math.ceil(len(data)/slen)

        for i in range(nbr_partition):

            self._compute_periodic_vars(data[i*slen:(i+1)*slen], cont_vars_part)

            for k, v in cont_vars_part.items():
                vars_means[k].append(np.mean(v))
                cont_vars_part[k] = []

        for k, v in vars_means.items():
            var = self[k]
            diff_means = []
            for i in range(len(v) - 1):
                for j in range(i+1, len(v)):
                    diff = math.fabs(v[i] - v[j])
                    diff_means.append(diff)
            same_dist_test = (var.max_val - var.min_val)*thresh
            self.vars[k].is_periodic = (np.mean(diff_means) <= same_dist_test)

    def continous_vars(self):
        return [x for x, j in self.items() if j.kind in [utils.HOL_REG, utils.INP_REG]]

    def discrete_vars(self):
        return [x for x, j in self.items() if j.kind in [utils.DIS_COIL, utils.DIS_INP]]

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

    data = utils.read_state_file("../SWaT_scripts/process_variables/SWat_Dataset_normal_v1.bin")[utils.COOL_TIME:]
    store = PVStore("./process_variables_limit_hist.yml", data)
    pdb.set_trace()
