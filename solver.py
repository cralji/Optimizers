from numpy import zeros,zeros_like,where,abs,minimum,concatenate
from numpy.linalg import eigvals,norm
from utils.pegasos import pegasos


from tensorflow import matmul,transpose

def proximal_point_subproblem(y,Qw,Qwc,rw,xw,xwc,gamma):
    obj = 0.5*transpose(y)@Qw@y+transpose(y)@(Qwc@xwc+rw)+0.5*gamma*matmul(y-xw,y-xw,transpose_a=True)
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
    
    def call(self, Q,r,l,u):
        kkt_res = 2*self.tol
        N,_ = Q.shape
        x = zeros((N,1))
        k = 0
        L = []
        U = []
        W = []
        Wc = []
        indx = list(range(N))
        
        while k<=self.max_iter and kkt_res>=self.tol:
            # asiggn sets 
            aux = Q.dot(x) + r
            L = where(aux<=-self.epsilon)[0].tolist()
            U = where(aux>=self.epsilon)[0].tolist()
            if self.epsilon==0:
                M = where(aux == self.epsilon)[0].tolist()
            else:
                M = where(abs(aux) <= self.epsilon)[0].tolist()
            W = [*set(L + U + M)] # unique values
            Wc = [i for i in indx if i not in W]
            W.sort() # ascending index sort
            Wc.sort() # ascending index sort
            Qw = Q[W][:,W]
            Qwc = Q[W][:,Wc]
            rw = r[W]
            xw = x[W]
            xwc = x[Wc]
            gamma = self.delta - minimum(0,eigvals(Qw).real.min())
            func = lambda y: proximal_point_subproblem(y,Qw,Qwc,rw,xw,xwc,gamma)
            sol_sub_problem = pegasos(func_obj=func,D=N,lam=gamma)
            y = sol_sub_problem.solver()
            x[W] = y
            ind_wc_l = [i for i in Wc if i in L]
            x[ind_wc_l] = l[ind_wc_l]
            ind_wc_u = [i for i in Wc if i in U]
            x[ind_wc_u] = u[ind_wc_u]

            kkt_res = compute_kkt_res(Q,x,l,u,r)
            k += 1
        