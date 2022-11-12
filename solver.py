#%% 
from numpy import zeros,zeros_like,where,abs,minimum,concatenate,all
from numpy.linalg import eigvals,norm
from numpy.random import uniform
from utils.pegasos import pegasos

import matplotlib.pyplot as plt

from tensorflow import matmul,transpose,Variable
from tensorflow.keras.optimizers import SGD

def proximal_point_subproblem(y,Qw,Qwc,rw,xw,xwc,gamma):
    if Qwc.shape[-1] ==0:
        obj= 0.5*transpose(y)@Qw@y+transpose(y)@(rw)+0.5*gamma*matmul(y-xw,y-xw,transpose_a=True)
    else:
        obj= 0.5*transpose(y)@Qw@y+transpose(y)@(Qwc@xwc+rw)+0.5*gamma*matmul(y-xw,y-xw,transpose_a=True)
    return obj

def compute_kkt_res(Q,x,l,u,r):
    aux = Q.dot(x) + r
    lambda_tilde = zeros_like(x)
    lambda_bar = zeros_like(x)
    ind1 = where(aux<0)[0].tolist()
    ind2 = where(aux>=0)[0].tolist()
    lambda_bar[ind1] = -aux[ind1]/(1+(u[ind1]-x[ind1])**2)
    lambda_tilde[ind2] = aux[ind2]/(1+(x[ind2]-l[ind2])**2)

    g = [aux-lambda_tilde+lambda_bar,
         minimum(x-l,0),
         minimum(u-x,0),
         minimum(lambda_tilde,0),
         minimum(lambda_bar,0),
         lambda_tilde*(x-l),
         lambda_bar*(u-x)]
    g = concatenate(g,axis=0)
    return norm(g)

def extract_sets(x,l,u):
    L = where(x <= l)[0]
    U = where(x >= u)[0]
    M = where((x < u)&(x > l))[0]
    return L,U,M
        

class RPPHom():
    """
        Solve the BoxConstraint problem:
        min 0.5*x'*Q*x + r'*x 
        s.t. l<= x <= u
    """
    def __init__(self,
                max_iter = 1000,
                tol = 1e-5,
                delta = 1e-2,
                epsilon = 1e-5):
        self.max_iter = max_iter
        self.tol = tol
        self.delta = delta
        self.epsilon = epsilon
    
    def call(self, Q,r,l,u,plot = False,verbose = False):
        kkt_res = 2*self.tol
        N,_ = Q.shape
        x = zeros((N,1))
        k = 0
        log_kkt_res = [kkt_res]
        indx = list(range(N))
        if all(x==l):
            x = uniform(l[0,0],u[0,0],N).reshape(-1,1)
        while k<=self.max_iter and kkt_res>=self.tol:
            # asiggn sets
            L,U,M = extract_sets(x, l, u)
            
            aux = Q.T.dot(x) + r
            U_ = U[where(aux[U]>=0)[0]].tolist() #self.epsilon
            L_ = L[where(aux[L]<=0)[0]].tolist() #-self.epsilon
            
            W = [*set(L_ + U_ + M.tolist())] # unique values
            Wc = [i for i in indx if i not in W]
            W.sort() # ascending index sort
            Wc.sort() # ascending index sort
            Qw = Q[W][:,W]
            rw = r[W]
            xw = x[W]
            Qwc = Q[W][:,Wc]
            xwc = x[Wc]
            gamma = self.delta - minimum(0,eigvals(Qw).real.min())
            # func = lambda y: proximal_point_subproblem(y,Qw,Qwc,rw,xw,xwc,gamma)
            # sol_sub_problem = pegasos(func_obj=func,D=Qwc.shape[0],lam=gamma)
            ### solution a proximal problem
            y = Variable(zeros([Qw.shape[0],1]))
            obj= lambda: 0.5*transpose(y)@Qw@y+transpose(y)@(Qwc@xwc+rw)+0.5*gamma*matmul(y-xw,y-xw,transpose_a=True)
            for t in range(1,1000):
                y_old = y.numpy()
                opt = SGD(learning_rate=1/(t*gamma))
                new_count = opt.minimize(obj,[y]).numpy()
                if norm(y_old-y.numpy()) < self.tol:
                    break
            y = y.numpy()
            
            # y = sol_sub_problem.solver()
            x[W] = y
            ind_wc_l = [i for i in Wc if i in L]
            x[ind_wc_l] = l[ind_wc_l]
            ind_wc_u = [i for i in Wc if i in U]
            x[ind_wc_u] = u[ind_wc_u]
            
            x[where(x<l)] = l[where(x<l)]
            x[where(x>u)] = u[where(x>u)]
            kkt_res = compute_kkt_res(Q,x,l,u,r)
            log_kkt_res.append(kkt_res)
            k += 1
            if verbose:
                print('k:{} ---- kkt_res:{}'.format(k,kkt_res))
            del y
            
        if plot:
            plt.plot(log_kkt_res)
            plt.show()
        if k < self.max_iter:
            self.converged = True
        else:
            self.converged = False
        self.x =x
        return x