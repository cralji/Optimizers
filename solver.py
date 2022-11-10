from utils.pegasos import pegasos
class RPPHom():
    """
        Solve the BoxConstraint problem:
        min 0.5*x'*Q*x + r'*x 
        s.t. l<= x <= u
    """
    def __init__(self,
                max_iter = 1000,
                tol = 1e-5,
                delta = 1e-2):
        self.max_iter = max_iter
        self.tol = tol
        self.delta = delta
    
    def call(self, Q,r,l,u):
        kkt_res = 2*self.tol
        N,_ = Q.shape
        for 
        