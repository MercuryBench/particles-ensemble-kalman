
import numpy as np
from collections import OrderedDict
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import smc_samplers as smp
from particles import kalman
from particles import resampling
from particles import mcmc
import particles

class RBMVLinearGauss(ssm.StateSpaceModel):
    """This is a class for a specific form of state space model of form
    .. math::
        W_0 & \sim PW0 \\
        Z_0|W_0 & \sim N(\mu_0,cov0(W_0)) \\
        W_t|W_{t-1} &\sim PW(W_{t-1}) \\
        Z_t|W_{t},Z_{t-1} &= A(W_{t}) Z_{t-1} + U_t, \quad U_t \sim N(a(W_{t}), Gamma(W_{t})) \\
        Y_t|W_{t},Z_t &= H(W_{t}) Z_t + V_t ,\quad V_t \sim N(h(W_{t}), Sigma(W_{t}))
    
    where F, G, C_U and C_V are matrices (depending on W_{t}).
    
    This means that there is a latent process Z_t, conditional on which everything becomes a linear State Space Model.
    Conditioning the variable Z with respect to observations Y is then done via Kalman filtering,
    with the marginalised likelihood of this Kalman step becoming the particle weight of a particle filter over W.

    The only mandatory parameters are `PW0`, `PW`, `Sigma` and `Gamma` (from which the
    dimensions dz and dy of, respectively, Z_t, and Y_t, are deduced). The
    default values for the other parameters are:

    * `mu0` : an array of zeros (of size dz)
    * `cov0`: cov_X
    * `F` : constant function returning identity matrix of shape (dz, dz).
    * `G` : constant function returning (dy, dz) matrix such that G[i, j] = 1[i=j]
    * `mU` : constant function returning 0-vector of shape (dz,) 
    * `mV` : constant function returning 0-vector of shape (dy,) 
    
    

    Optionally, we may also define methods:

    * `proposal0(self, data)`: the (data-dependent) proposal dist at time 0
    * `proposal(self, t, xp, data)`: the (data-dependent) proposal distribution at
      time t, for X_t, conditional on X_{t-1}=xp
    * `logeta(self, t, x, data)`: the auxiliary weight function at time t

    You need these extra methods to run a guided or auxiliary particle filter.

    """

    def __init__(self, PW0=None, PW=None, A=None, H=None, Gamma=None, Sigma=None, m0=None, cov0=None, a=None, h=None):
        self.w0 = PW0.rvs(1) # test PW0 and create an element for testing dimensions
        self.Gamma, self.Sigma = Gamma, Sigma
        self.dz, self.dy = self.Gamma(self.w0).shape[0], self.Sigma(self.w0).shape[0]
        self.m0 = (lambda th: np.zeros(self.dz)) if m0 is None else m0
        self.cov0 = self.Gamma if cov0 is None else cov0
        self.A = (lambda th: np.eye(self.dz)) if A is None else A # TODO: make this also a default if A is not callable
        self.H = (lambda th: np.eye(self.dy, self.zx)) if H is None else H
        self.a = (lambda th: np.zeros(self.dz)) if a is None else a
        self.h = (lambda th: np.zeros(self.dy)) if h is None else h
        self.PW0 = PW0
        self.PW = PW

    def _error_msg(self, method):
        return (
            "method " + method + " not implemented in class%s" % self.__class__.__name__
        )

    def check_shapes(self):
        """
        Check all dimensions are correct.
        """
        assert self.covX(self.theta0).shape == (self.dz, self.dz), error_msg
        assert self.covY(self.theta0).shape == (self.dy, self.dy), error_msg
        assert self.F(self.theta0).shape == (self.dz, self.dz), error_msg
        assert self.G(self.theta0).shape == (self.dy, self.dz), error_msg
        assert self.mu0(self.theta0).shape == (self.dz,), error_msg
        assert self.mU(self.theta0).shape == (self.dz,), error_msg
        assert self.mV(self.theta0).shape == (self.dy,), error_msg
        assert self.cov0(self.theta0).shape == (self.dz, self.dz), error_msg
        assert self.PW0(self.theta0).rvs(1).shape == (self.dz)


    @classmethod
    def state_container(cls, N, T):
        law_X0 = cls().PX0()
        dim = law_X0.dim
        shape = [N, T]
        if dim > 1:
            shape.append(dim)
        return np.empty(shape, dtype=law_X0.dtype)


    def PZ0W0(self, w0): #Z_0|W_0 & \sim N(\mu_0,cov0(W_0)) 
        return dists.MvNormal(loc=self.m0(w0), cov=self.cov0(w0))
    
    def PX0(self):
        chain_rule = OrderedDict()
        chain_rule['w'] = self.PW0
        chain_rule['z'] = dists.Cond(lambda th: self.PZ0W0(th['w']), dim=self.dz)
        return dists.StructDist(chain_rule, force_sequentially=True)
        
    def PZ(self, t, zp, w): #Z_t|W_{t},Z_{t-1} &= F(W_{t}) Z_{t-1} + U_t, \quad U_t \sim N(m_U(W_{t}), covZ(W_{t})) \\
        return dists.MvNormal(loc=np.dot(zp, self.A(w).T) + self.a(w), cov=self.Gamma(w))

    def PY(self, t, xp, x): #Y_t|W_{t},Z_t &= G(W_{t}) Z_t + V_t ,\quad V_t \sim N(m_V(W_{t}), covY(W_{t}))
        w = x['w']
        z = x['z']
        if len(z) == 1:
            return dists.MvNormal(loc=np.dot(z, self.H(w).T) + self.h(w), cov=self.Sigma(w))
        else:
            k_distr = [dists.MvNormal(loc=np.dot(z[k], self.H(w[k]).T) + self.h(w[k]), cov=self.Sigma(w[k])) for k in range(len(z))]
            return dists.IndepProd(*k_distr, variant=True)
    
    def PX(self, t, xp):
        wp = xp['w']
        zp = xp['z']
        if len(xp) == 1: #np.isscalar(zp):
            wp = wp[0]
            zp = zp[0]
            chain_rule = OrderedDict()
            chain_rule['w'] = self.PW(t, wp)
            chain_rule['z'] = dists.Cond(lambda th: self.PZ(t, zp, th['w']), dim=self.dz)
            return dists.StructDist(chain_rule, force_sequentially=True)
        else:                
            chain_rule = [OrderedDict() for k in range(len(zp))]
            for k in range(len(zp)):
                chain_rule[k]['w'] = self.PW(t, wp[k])
                chain_rule[k]['z'] = dists.Cond(lambda th: self.PZ(t, zp[k], th['w']))
        return dists.IndepProd(*[dists.StructDist(chain_rule[k]) for k in range(len(zp))], variant=True)
        
        

    def proposal0(self, data):
        raise NotImplementedError(self._error_msg("proposal0"))

    def proposal(self, t, xp, data):
        """Proposal kernel (to be used in a guided or auxiliary filter).

        Parameter
        ---------
        t: int
            time
        x:
            particles
        data: list-like
            data
        """
        raise NotImplementedError(self._error_msg("proposal"))



class Bootstrap_RaoBlackwell(particles.FeynmanKac):
      
  # default_params = {'gamma' : 0.1, #"D0": 0.1, "Drot": 1.0,"Dobs": 0.01,  
  #                   'm0': np.array([0]), 'cov0': np.diag([0.01])}
  def __init__(self, ssm=None, data=None, **kwargs):
    self.ssm = ssm
    self.data = data
    self.T = 0 if data is None else len(data)
    
    # self.F = None
    # self.P = None
    
    # self.A = None
    # # self.a = None # TODO: implement added constant in Kalman
    # self.Gamma = None
    
    # self.H = None
    # # self.b = None # TODO: implement added constant in Kalman
    # self.Sigma = None
    # self.a = lambda th: None
    # self.h = lambda th: None
    
    # self.PW0 = None
    # self.PW = None
    
    # self.m0 = None
    # self.cov0 = None
    
    # self.__dict__.update(self.default_params)  
    self.__dict__.update(kwargs)
    
    # define PX0
    # self.PW0 = dists.Categorical(p = np.array([0.5, 0.5]))
    # chain_rule = OrderedDict()
    # chain_rule['w'] = self.PW0
    # chain_rule['z'] = dists.Cond(lambda x: dists.MvNormal(loc=self.m0(x['w']), cov=self.cov0(x['w'])))
    # self.PX0 = dists.StructDist(chain_rule, force_sequentially=True)
    
    # def PW(wp):  # The law of W_t conditional on W_{t-1} # TODO: vectorise
    #     p_transition = (wp == 0)[:,np.newaxis]*np.array([1-gamma, gamma]) + (wp == 1)[:,np.newaxis]*np.array([gamma, 1-gamma])  # switch with prob. gamma
    #     return dists.Categorical(p = p_transition)
    # self.PW = PW
  
  def M0(self, N):
    ws = self.ssm.PW0.rvs(size=N)
    MC0s = smp.FancyList([kalman.MeanAndCov(mean=self.ssm.m0(ws[j]), cov=self.ssm.cov0(ws[j])) for j in range(N)])
    
    
    # do all Kalman predictions
    J = N
    # MC_pred = [None for j in range(J)]
    MCs_filter = [None for j in range(J)]
    logpyts = np.zeros(J)#np.array([0.0 for j in range(J)])
    for j in range(J):
      MCs_filter[j], logpyts[j] = kalman.filter_step(self.ssm.H(ws[j]), self.ssm.Sigma(ws[j]), MC0s[j], self.data[0])
    return smp.ThetaParticles(w=ws, MC=smp.FancyList(MCs_filter), logGs=logpyts)
  
  def M(self, t, xp):
    J = xp.N
    ws_old = xp.w
    ws_new = self.ssm.PW(t, ws_old).rvs(J)    
    
    # do all Kalman predictions
    MC_pred = [None for j in range(J)]
    MC_filter = [None for j in range(J)]
    logpyts = np.zeros(J)
    for j, part in enumerate(xp):
      MC_old = part['MC']
      MC_pred[j] = kalman.predict_step(self.ssm.A(part['w']), self.ssm.Gamma(part['w']), MC_old, self.ssm.a(part['w']))
      MC_filter[j], logpyts[j] = kalman.filter_step(self.ssm.H(part['w']), self.ssm.Sigma(part['w']), MC_pred[j], self.data[t], self.ssm.h(part['w']))
    return smp.ThetaParticles(w=ws_new, MC=smp.FancyList(MC_filter), logGs=logpyts)  

  def logG(self, t, xp, x):
    # all these computations have been done in a previous M step, so in principle just load them
    return x.logGs

class RBMVLinearGauss_1d(RBMVLinearGauss):
    """ Special case:
        ############################################################### edit from here
            class LinearGaussDirection(RBLinearStateSpaceModel):
                def PX0(self):  # The law of W_0
                    return dists.Categorical(p = np.array([0.5, 0.5]))
                def PZ0W0(self, w0): # The law of Z_0 conditional on W0, here independent
                    return dists.Normal(loc=self.mu0, scale=self.sigma0)
                def PW(self, t, wp):  # The law of W_t conditional on W_{t-1} # TODO: vectorise
                    p_transition = (wp == 0)*np.array([1-self.gamma, self.gamma]) + (wp == 1)*np.array([self.gamma, 1-self.gamma])  # switch with prob. gamma
                    return dists.Categorical(p = p_transition)
                def PZW(self, t, zp, wp, w):  # the law of Z_t given Z_{t-1} and W_{t-1} and W_t
                    return dists.Normal(loc=zp+self.h*wp, scale=self.sigmaZ)
                def PY(self, t, zp, z, wp, w):  # the law of Y_t given Z_{t-1}, Z_t, W_{t-1} and W_t
                    return dists.Normal(loc=z, scale=self.sigmaY) # CAUTION: DOES NOT DEFINE THE MATRICES FOR KALMAN

        These methods return ``ProbDist`` objects, which are defined in the module
        `distributions`. The model above is a basic linear Gaussian SSM; it
        depends on parameters rho, sigmaX, sigmaY (which are attributes of the
        class). To define a particular instance of this class, we do::

            #a_certain_ssm = LinearGauss(rho=.8, sigmaX=1., sigmaY=.2)
            a_certain_rbssm = LinearGaussDirection(mu0 = 0., sigma0 = 1., gamma = 0.1, h=0.1, sigmaZ=0.2, sigmaY=0.5)

        All the attributes that appear in ``PX0``, ``PX`` and ``PY`` must be
        initialised in this way. Alternatively, it it possible to define default
        values for these parameters, by defining class attribute
        ``default_params`` to be a dictionary as follows::

            class LinearGauss(StateSpaceModel):
                default_params = {'rho': .9, 'sigmaX': 1., 'sigmaY': .1}
                # rest as above """
    def __init__(self, gamma = 0.1, h=0.1, mu0 = 0., sigma0 = 1., sigmaZ = 0.2, sigmaY = 0.5):
        PW0 = dists.Categorical(p = np.array([0.5, 0.5]))
        def PW(self, wp):  # The law of W_t conditional on W_{t-1} # TODO: vectorise
            p_transition = (wp == 0)*np.array([1-self.gamma, self.gamma]) + (wp == 1)*np.array([self.gamma, 1-self.gamma])  # switch with prob. gamma
            return dists.Categorical(p = p_transition)
        F = lambda th: np.array([[1]])
        G = lambda th: np.array([[1]])
        covZ = lambda th: np.array([[sigmaZ**2]])
        covY = lambda th: np.array([[sigmaY**2]])
        mu0 = lambda th: np.array([0])
        cov0 = lambda th: np.array([[sigma0**2]])
        mU = lambda th: h*th # TODO: check dimensions
        mV = lambda th: np.array([0])
        RBMVLinearGauss.__init__(self,  PW0=PW0, PW=PW, F=F, G=G, covZ=covZ, covY=covY, mu0=mu0, cov0=cov0, mU=mU, mV=mV)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # build prior
    
    # np.random.normal(np.array([1,20]), size=2)
    
    # chain_rule = OrderedDict()
    # chain_rule['mu'] = dists.Categorical(p = np.array([0.5, 0.5]))
    # chain_rule['tau'] = dists.Cond(lambda x: dists.Normal(loc=x['mu'], scale=0.001))
    # prior = dists.StructDist(chain_rule)
    # prior.rvs(20)
    
    # pp = dists.MvNormal()
    
    # chain_rule = OrderedDict()
    # chain_rule['mu'] = dists.MvNormal(loc = np.array([0., 0.]))
    # chain_rule['tau'] = dists.Cond(lambda x: dists.MvNormal(loc=x['mu'], scale=0.001), dim=2)
    # prior = dists.StructDist(chain_rule)
    # prior.rvs(2)
    
    
    # chain_rule = OrderedDict()
    # chain_rule['w'] = dists.Categorical(p = np.array([0.5, 0.5]))
    
    # chain_rule['z'] = dists.Cond(lambda x: dists.MvNormal(loc=x['w']*np.array([1,0]), cov=np.array([[0.01, 0],[0,0.01]])), dim=2)
    # mydist = dists.StructDist(chain_rule)
    # mydist.rvs(5, force_sequentially=True)
    
    # J = 5
    # chain_rule = OrderedDict()
    # chain_rule['w'] = dists.Categorical(p = np.array([0.5, 0.5]))
    # #chain_rule['z'] = dists.Cond(lambda x: dists.Normal(loc=x['w'], scale=0.01), dim=1)
    # chain_rule['z'] = dists.Cond(lambda x: dists.MvNormal(loc=x['w']*np.array([1]), scale=0.01), dim=1)
    # PX0 = dists.StructDist(chain_rule)
    # x0 = PX0.rvs(J, force_sequentially=True)
    # w0 = x0['w']
    # z0 = x0['z']
    
    # gamma = 0.1
    # def PW(wp):  # The law of W_t conditional on W_{t-1} # TODO: vectorise
    #     p_transition = (wp == 0)[:,np.newaxis]*np.array([1-gamma, gamma]) + (wp == 1)[:,np.newaxis]*np.array([gamma, 1-gamma])  # switch with prob. gamma
    #     return dists.Categorical(p = p_transition)
    # w1 = PW(x0['w']).rvs()
    
    # h = 1.0
    # F = lambda w: np.array([[1]])
    # mU = lambda w: h*w
    # covZ = lambda w: 0.01
    # def PZW(t, zp, w): #Z_t|W_{t},Z_{t-1} &= F(W_{t}) Z_{t-1} + U_t, \quad U_t \sim N(m_U(W_{t}), covZ(W_{t})) \\            
    #     return dists.MvNormal(loc=np.dot(zp, F(w).T) + mU(w), cov=np.array([[covZ(w)]]))
    
    # z1 = np.array([PZW(0, z0[k], w1[k]).rvs() for k in range(J)]).flatten()
    
    
    
    # RBMVLinearGauss(ssm.StateSpaceModel):
       
    case = 3

    if case == 1: 
        #     def __init__(self, PW0=None, PW=None, A=None, H=None, Gamma=None, Sigma=None, m0=None, cov0=None, a=None, h=None):
        PW0 = dists.Categorical(p = np.array([0.5, 0.5]))
        def PW(t, wp, gamma=0.1):  # The law of W_t conditional on W_{t-1}
            if np.isscalar(wp):
                p_transition = (wp == 0)*np.array([1-gamma, gamma]) + (wp == 1)*np.array([gamma, 1-gamma])  # switch with prob. gamma
            else:
                p_transition = (wp[:,np.newaxis] == 0)*np.array([1-gamma, gamma]) + (wp[:,np.newaxis] == 1)*np.array([gamma, 1-gamma])  # switch with prob. gamma
                
            return dists.Categorical(p = p_transition)
        A = lambda w: np.array([[1.]])
        H = lambda w: np.array([[1.]])
        Gamma = lambda w: np.array([[0.01]])
        Sigma = lambda w: np.array([[0.01]])
        m0 = lambda w: np.array([0.])
        cov0 = lambda w: np.array([[1.]])
        a = lambda w: w
        h = lambda w: np.array([0])
        myRB = RBMVLinearGauss(PW0, PW, A, H, Gamma, Sigma, m0, cov0, a, h)
        J = 5
        # x0s = myRB.PX0().rvs(J)
        # x1s = np.empty(shape=x0s.shape, dtype=x0s.dtype)
        # myRB.PX(0, x0s[0]).rvs(1)
        #x1s = np.array([myRB.PX(0, x0s[k]).rvs(1) for k in range(J)], dtype=x0s.dtype).flatten()
        # x1s = myRB.PX(0, x0s).rvs(J)
        xs, ys = myRB.simulate(10)
        xs = np.vstack(xs)
        ys = np.vstack(ys).flatten()
        plt.figure()
        plt.plot(xs['w'], '.-', label="w")
        plt.plot(xs['z'], '.-', label="z")
        plt.plot(ys, '.-', label="y")
        plt.legend()
        
        mybootstrap = ssm.Bootstrap(myRB, ys)
        x0 = mybootstrap.M0(5)
        x1 = mybootstrap.M(0, x0)
        # mybootstrap.logG(0, x0, x1)
        ######### DIRECTLY DEFINE FK object next, using thetaparticles and m,C
        # myRB = RBMVLinearGauss_1d()
        # myRB.PX0().rvs(5)
        # myRB.simulate(10)
        # my_FKRB = FK_ABM_RB_Bootstrap(data = data)
        # pf = particles.SMC(fk=my_FKRB, N=1000, store_history=True)
        # # plot_thetaPart(pf.X, pf.wgts, 0)  
        # pf.run()
        
        #myRB = RBMVLinearGauss_1d()
        #myRB.PX0().rvs(5)
        #myRB.simulate(10)
        
        myBootRB = Bootstrap_RaoBlackwell(data = ys, A=A, Gamma=Gamma, H=H, Sigma=Sigma, a=a, h=h, PW0=PW0, PW=PW, m0=m0, cov0=cov0)
        tp0 = myBootRB.M0(5)
        print(tp0.MC[:])
        
        pf = particles.SMC(fk=myBootRB, N=5, store_history=True)
        pf.run()
         
    elif case == 2:
        # 2nd example
        from math import sqrt, cos, sin, pi
        D0 = 1.4
        Drot = 0.5
        Dobs = 0.1
        dt = 0.1
        # A = lambda w: np.array([[1., 0., dt*cos(w)],[0., 1., dt*sin(w)],[0.,0.,1.]])
        # H = lambda w: np.array([[1,0,0],[0,1,0]])
        # Gamma = lambda w: np.diag([(2*dt*D0),(2*dt*D0),0.00001])
        # Sigma = lambda w: np.eye(2)*Dobs
        # m0 = lambda w: np.array([0.,0.,20.])
        # cov0 = lambda w: np.diag([5.,5.,10.])
        # PW0 = dists.Uniform(0, 2*pi)
        # def PW(t, wp):
        #     return dists.Normal(loc=wp, scale=sqrt(Drot))
        # myRB_nd = RBMVLinearGauss(PW0, PW, A, H, Gamma, Sigma, m0, cov0)
        
        class ABM(RBMVLinearGauss):
            def __init__(self, **kwargs):
                default_parameters = {'D0': D0, 'Drot': Drot, 'Dobs': Dobs}
                self.__dict__.update(default_parameters)
                self.__dict__.update(kwargs)
                A = lambda w: np.array([[1., 0., dt*cos(w)],[0., 1., dt*sin(w)],[0.,0.,1.]])
                H = lambda w: np.array([[1,0,0],[0,1,0]])
                Gamma = lambda w: np.diag([(2*dt*self.D0),(2*dt*self.D0),0.00001])
                Sigma = lambda w: np.eye(2)*self.Dobs
                m0 = lambda w: np.array([0.,0.,20.])
                cov0 = lambda w: np.diag([5.,5.,10.])
                PW0 = dists.Uniform(0, 2*pi)
                def PW(t, wp):
                    return dists.Normal(loc=wp, scale=sqrt(self.Drot))
                RBMVLinearGauss.__init__(self,PW0, PW, A, H, Gamma, Sigma, m0, cov0)
                
        
                
        # class FK_ABM(Bootstrap_RaoBlackwell):
        #     def __init__(self, **kwargs):
        #         default_parameters = {'D0': D0, 'Drot': Drot, 'Dobs': Dobs}
        #         self.__dict__.update(default_parameters)
        #         self.__dict__.update(kwargs)
        #         A = lambda w: np.array([[1., 0., dt*cos(w)],[0., 1., dt*sin(w)],[0.,0.,1.]])
        #         H = lambda w: np.array([[1,0,0],[0,1,0]])
        #         Gamma = lambda w: np.diag([(2*dt*self.D0),(2*dt*self.D0),0.00001])
        #         Sigma = lambda w: np.eye(2)*self.Dobs
        #         m0 = lambda w: np.array([0.,0.,20.])
        #         cov0 = lambda w: np.diag([5.,5.,10.])
        #         PW0 = dists.Uniform(0, 2*pi)
        #         def PW(t, wp):
        #             return dists.Normal(loc=wp, scale=sqrt(self.Drot))
        #         Bootstrap_RaoBlackwell.__init__(self, data = ys, A=A, Gamma=Gamma, H=H, Sigma=Sigma, PW0=PW0, PW=PW, m0=m0, cov0=cov0)
        # myBootRB = Bootstrap_RaoBlackwell(myRB_nd, ys)#FK_DiffDiff(myRB_nd, ys)
        # # results = particles.multiSMC(fk=myBootRB, N=500, nruns=30)
        # # plt.figure()
        # # plt.boxplot([r['output'].logLt for r in results]);
        # pf = particles.SMC(fk=myBootRB, N=1000, store_history=True)
        # pf.run()
        myRB_nd = ABM()
        xs, ys = myRB_nd.simulate(10)
        xs = np.vstack(xs).flatten()
        ys = np.vstack(ys)
        
        plt.figure()
        plt.plot(xs['z'][:,0], xs['z'][:,1], '.-')
        plt.plot(ys[:,0], ys[:,1], '.-')
        plt.axis("equal")
        import seaborn as sb
        #myBootRB = Bootstrap_RaoBlackwell(data = ys, A=A, Gamma=Gamma, H=H, Sigma=Sigma, PW0=PW0, PW=PW, m0=m0, cov0=cov0)
        
        
        # myBootRB = FK_ABM()
        myBootRB = Bootstrap_RaoBlackwell(myRB_nd, ys)
        results = particles.multiSMC(fk=myBootRB, N=500, nruns=30)
        plt.figure()
        plt.boxplot([r['output'].logLt for r in results]);
        pf = particles.SMC(fk=myBootRB, N=1000, store_history=True)
        pf.run()
        
        print(pf.logLt)
        
        def plot_thetaPart(tp, wgts, k=None, N_smp=50):
            angles = tp.w
            ms = [MC.mean for MC in tp.MC]
            Cs = [MC.cov for MC in tp.MC]
            
            plt.figure(figsize=(3,6))
            plt.subplot(311)
            plt.hist(np.mod(angles, 2*pi),50,weights=wgts.W)
            if k is not None:
                plt.axvline(np.mod(xs['w'][k], 2*pi), color='k')
                #plt.axvline(angles[k], color='k')
            plt.xlim([-0.5,2*pi+0.5])
            plt.title("phi")
            
            indices = resampling.stratified(wgts.W)
            
            samples = np.vstack([np.random.multivariate_normal(ms[0].flatten(), Cs[0], N_smp) for k in indices])
            
            plt.subplot(312)
            # plt.plot(samples[:,0], samples[:, 1], '.', alpha=0.01)
            plt.hist2d(samples[:,0], samples[:, 1], 20)
            if k is not None:
                plt.plot(xs['z'][k][0], xs['z'][k][1], 'kx')
                plt.plot(ys[k][0], ys[k][1], 'rx')
            plt.title("x,y")
            
            plt.subplot(3,1,3)
            plt.hist(samples[:,2], 20)
            if k is not None:
                plt.axvline(xs['z'][0,2], color='k')
            plt.title("v")
            plt.xlim([-30,30])
            plt.tight_layout()
        
        plot_thetaPart(pf.X, pf.wgts, k=len(ys)-1)
        
        case = "Drot_D0"
        
        if case == "Drot_D0_Dobs":
            prior_dict = {'D0': dists.Gamma(),
                  'Drot': dists.Gamma(),
                  'Dobs':dists.Gamma(a=1., b=5)}
            true_vals = [D0, Drot, Dobs]
        else:
            prior_dict = {'D0': dists.Gamma(),
                  'Drot': dists.Gamma()}
            true_vals = [D0, Drot]
        my_prior = dists.StructDist(prior_dict)
        
        my_pmmh = mcmc.PMMH(ssm_cls=ABM, fk_cls=Bootstrap_RaoBlackwell, prior=my_prior, data=ys, Nx=500,  niter=1000, verbose=100)
        my_pmmh.run()
        for mm, p in enumerate(prior_dict.keys()):  # loop over D0, Drot, Dobs
            plotrange = np.linspace(prior_dict[p].ppf(0.05), prior_dict[p].ppf(0.85), 100)
            plt.figure()
            plt.subplot(211)
            plt.plot(my_pmmh.chain.theta[p], label="samples")
            plt.axhline(true_vals[mm], color="tab:orange", label="true")
            plt.xlabel('iter')
            plt.ylabel(p)
            plt.title("samples over time")
            plt.subplot(212)
            plt.hist(my_pmmh.chain.theta[p], 150, range=(plotrange[0],plotrange[-1]), density=True, label="samples")
            plt.axvline(true_vals[mm], color="tab:orange", label="true")
            plt.plot(plotrange, prior_dict[p].pdf(plotrange), '--', color="tab:green", label="prior density")
            plt.xlabel(p)
            plt.title("histogram of samples")
            plt.legend()
            #plt.xlim()
            plt.tight_layout()
        
        array_samples = np.stack([my_pmmh.chain.theta[p] for p in prior_dict.keys()])
        import corner
        corner.corner(array_samples.T, truths=true_vals, labels=[p for p in prior_dict.keys()])
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.hist(np.mod(pf.X.w, 2*pi), 50, weights=pf.wgts.W)
        # plt.axvline(xs['w'][-1], color="tab:orange")
        # plt.subplot(2,1,2)
    else:
        def plot_thetaPart(tp, wgts, k=None, N_smp=50):
            diffs = tp.w
            ms = [MC.mean for MC in tp.MC]
            Cs = [MC.cov for MC in tp.MC]
            
            plt.figure(figsize=(3,6))
            plt.subplot(211)
            plt.hist(diffs,50,weights=wgts.W)
            if k is not None:
                plt.axvline(xs['w'][k], color='k')
                #plt.axvline(angles[k], color='k')
            plt.title("phi")
            
            indices = resampling.stratified(wgts.W)
            
            samples = np.vstack([np.random.multivariate_normal(ms[0].flatten(), Cs[0], N_smp) for k in indices])
            
            plt.subplot(212)
            # plt.plot(samples[:,0], samples[:, 1], '.', alpha=0.01)
            plt.hist2d(samples[:,0], samples[:, 1], 20)
            if k is not None:
                plt.plot(xs['z'][k][0], xs['z'][k][1], 'kx')
                plt.plot(ys[k][0], ys[k][1], 'rx')
            plt.title("x,y")
            
            plt.tight_layout()
        dt = 0.05
        m_param = 3.0
        tau_param = 0.5
        sigma_param= 1.0
        Dobs = 0.01
        from math import sqrt
        class DiffDiff(RBMVLinearGauss):
            def __init__(self, **kwargs):
                default_parameters = {'m_param': m_param, 'tau_param': tau_param, 'sigma_param': sigma_param}
                self.__dict__.update(default_parameters)
                self.__dict__.update(kwargs)
                A = lambda w: np.eye(2)
                H = lambda w: np.eye(2)
                Gamma = lambda w: np.eye(2)*2*dt*w
                Sigma = lambda w: np.eye(2)*2*dt*sqrt(Dobs)
                m0 = lambda w: np.zeros(2)
                cov0 = lambda w: np.diag([5.,5.])
                PW0 = dists.Gamma()
                def PW(t, wp):
                    mean_D = wp +1/self.tau_param*(self.m_param - wp)*dt
                    scale_D = self.sigma_param*np.sqrt(2*wp*dt)
                    return dists.TruncNormal(mu=mean_D, sigma=scale_D, a=0.0, b=1000.0)
                RBMVLinearGauss.__init__(self,PW0, PW, A, H, Gamma, Sigma, m0, cov0)
        # class FK_DiffDiff(Bootstrap_RaoBlackwell):
        #     def __init__(self, **kwargs):
        #         default_parameters = {'m_param': m_param, 'tau_param': tau_param, 'sigma_param': sigma_param}
        #         self.__dict__.update(default_parameters)
        #         self.__dict__.update(kwargs)
        #         # A = lambda w: np.eye(2)
        #         # H = lambda w: np.eye(2)
        #         # Gamma = lambda w: np.eye(2)*2*dt*w
        #         # Sigma = lambda w: np.eye(2)*2*dt*w
        #         # m0 = lambda w: np.zeros(2)
        #         # cov0 = lambda w: np.diag([5.,5.])
        #         # PW0 = dists.Gamma()
        #         # def PW(t, wp):
        #         #     mean_D = wp +1/self.tau_param*(self.m_param - wp)*dt
        #         #     scale_D = self.sigma_param*np.sqrt(2*wp*dt)
        #         #     return dists.TruncNormal(mu=mean_D, sigma=scale_D, a=0.0, b=1000.0)
        #         Bootstrap_RaoBlackwell.__init__(self, data = ys, A=A, Gamma=Gamma, H=H, Sigma=Sigma, PW0=PW0, PW=PW, m0=m0, cov0=cov0)
        myRB_nd = DiffDiff()
        N_sim = 200
        
        ts = np.arange(0,dt*N_sim,dt)
        
        xs, ys = myRB_nd.simulate(N_sim)
        xs = np.vstack(xs).flatten()
        ys = np.vstack(ys)
        
        plt.figure()
        plt.plot(xs['z'][:,0], xs['z'][:,1], '.-')
        plt.plot(ys[:,0], ys[:,1], '.-')
        plt.axis("equal")
        
        plt.figure()
        plt.subplot(211)
        plt.plot(ts, xs['w'], label='diff')
        plt.legend()
        plt.subplot(212)
        plt.plot(ts, xs['z'][:,0],label='x')
        plt.plot(ts, xs['z'][:,1],label='y')
        plt.legend()
        
        myBootRB = Bootstrap_RaoBlackwell(myRB_nd, ys)#FK_DiffDiff(myRB_nd, ys)
        # results = particles.multiSMC(fk=myBootRB, N=500, nruns=30)
        # plt.figure()
        # plt.boxplot([r['output'].logLt for r in results]);
        pf = particles.SMC(fk=myBootRB, N=1000, store_history=True)
        pf.run()
        
        # for nn in range(N_sim):
        #     plot_thetaPart(pf.hist.X[nn], pf.hist.wgts[nn], k=nn)
        
        prior_dict = {'m_param': dists.Uniform(0.0, 5.0),
              'tau_param': dists.Uniform(0.0, 5.0),
              'sigma_param':dists.Uniform(0.0, 5.0)}
        true_vals = [m_param, tau_param, sigma_param]
        my_prior = dists.StructDist(prior_dict)
        
        my_pmmh = mcmc.PMMH(ssm_cls=DiffDiff, fk_cls=Bootstrap_RaoBlackwell, prior=my_prior, data=ys, Nx=250,  niter=4000, verbose=1000)
        my_pmmh.run()
        
        for mm, p in enumerate(prior_dict.keys()):  # loop over D0, Drot, Dobs
            plotrange = np.linspace(prior_dict[p].ppf(0.001), prior_dict[p].ppf(0.999), 100)
            plt.figure()
            plt.subplot(211)
            plt.plot(my_pmmh.chain.theta[p], label="samples")
            plt.axhline(true_vals[mm], color="tab:orange", label="true")
            plt.xlabel('iter')
            plt.ylabel(p)
            plt.title("samples over time")
            plt.subplot(212)
            plt.hist(my_pmmh.chain.theta[p], 150, range=(plotrange[0],plotrange[-1]), density=True, label="samples")
            plt.axvline(true_vals[mm], color="tab:orange", label="true")
            plt.plot(plotrange, prior_dict[p].pdf(plotrange), '--', color="tab:green", label="prior density")
            plt.xlabel(p)
            plt.title("histogram of samples")
            plt.legend()
            #plt.xlim()
            plt.tight_layout()
        
        array_samples = np.stack([my_pmmh.chain.theta[p] for p in prior_dict.keys()])
        import corner
        corner.corner(array_samples.T, truths=true_vals, labels=[p for p in prior_dict.keys()])
        

