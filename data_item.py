class data_item(object):
    def __init__(self, vals):
        self.label = vals[-1]
        if len(vals) > 1:
            self.attributes = list(map(int, vals[0:len(vals) - 1]))

    def __str__(self):
        return str(self.attributes) + " " + str(self.label)