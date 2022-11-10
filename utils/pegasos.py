#%%
from tensorflow import GradientTape,Variable,zeros,norm
import matplotlib.pyplot as plt


import tensorflow as tf


class pegasos():
    def __init__(self,
                func_obj,
                D = 2,
                lam=1,
                T = 1000,
                tol = 1e-4,
                plot = False):
        self.func_obj = func_obj
        self.D = D
        self.lam = lam
        self.T = T
        self.tol = tol
        self.plot = plot
    
    def solver(self):
        error = []
        y = Variable(zeros([self.D,1])) if self.D!=1 else Variable(0.0)
        for t in range(1,1+self.T):
            eta_t = 1/(self.lam*t)
            with GradientTape() as tape:
                obj = self.func_obj(y)
            grad = tape.gradient(obj,y)
            y_new = y - eta_t*grad
            print(y.numpy(),y_new.numpy(),)
            error.append(norm(y.numpy()-y_new.numpy()))
            # print(y.numpy(),y_new.numpy(),error[-1])
            y.assign(y_new)
            if error[-1] < self.tol:
                break
        if self.plot:
            plt.plot(range(1,1+len(error)), error,label = r'$||y-y_{new}||$')
            plt.legend()
            plt.show()
        return y.numpy()