import numpy as np


"""
Given set of points that may fit the equation y = a*e^bx + c
"""

class GradDescent:

    def __init__(self, lr = 0, iterations=100, x, y ):
        self.lr = 0.2
        self.iterations =iterations
        self.params = None
        self.x = x
        self.y = y
    def init_params(self):
        
        return np.array([1.0, 1.0, 1.0])

    def gradient(self, x, y, params):

        a = params[0]
        b = params[1]
        c = params[2]
        y_pred = a*np.exp(b*x) + c
        error = y_pred - y

        da = b*np.exp(b*x)
        db = a*np.exp(a*x*np.exp(b*x))
        dc = 1

        return [da, db, dc]
    
    def descent(self,):
        params = self.init_params()
        params = self.init_params()
        for i in len(self.iterations):
            grd = self.gradient(x, y, params)
            params = -self.lr*()*self.gradient

        return params

        









