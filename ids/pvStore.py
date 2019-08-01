import yaml
import utils

class PVStore(object):

    def __init__(self, descFile):
        # name -> Process Variable
        self.vars = {}
        self.setup(descFile)

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
