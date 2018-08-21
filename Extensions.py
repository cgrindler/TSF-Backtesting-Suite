import Specification

class Extensions:
    def __init__(self):
        self.Specs = Specification.Specification
        self.loglevel = self.Specs.loglevel

    def Log(self,message, level = 1):
        if level>=self.loglevel:
            print(message)
