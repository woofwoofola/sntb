#  hyperparameter_bayesian_priors.py
# Copyright 2020 André Carrington, Ottawa Hospital Research Institute
# Use is subject to the Apache 2.0 License
# Written by André Carrington
#
#  define the distribution of each hyperparameter
#  using Hyperopt (hp) distributions 
#  the corresponding classifiers are denoted in comments afterward


# knn hyperparameters
k        =   hp.choice('k', np.arange(1, 26, 2, dtype=int)) # knn; k=1,3,5,...,25


# logr, plogr1, plogr2, plogrE1, plogrE2, svc, nsvc, svr hyperparameters
# C      = regularization parameter (logr)
# C      = box constraint or cost of errors (svc, svr)
#
# note: a simplistic distribution for C, as follows, is avoided...
# C      =   hp.uniform('svc_C',     0, 1000   ))      # default=1.0
#
# rationale:
#   some high values of C hang or do not converge
#   some values of C are less likely to apply or do well (so use them less)
#
# hence we experiment with multiple distributions: C, C2, C3, C4, C5...
C        =   hp.pchoice('C',
      [(.75, hp.uniform('Csmall'   ,0.03 ,5    )),
       (.15, hp.uniform('Cmed'     ,5    ,100  )),
       (.1 , hp.uniform('Chigh'    ,100  ,4000 ))])

C2       =   hp.pchoice('C2',
      [(.75, hp.uniform('Csmall2'  ,0.001,20   )),
       (.15, hp.uniform('Cmed2'    ,20   ,100  )),
       (.10, hp.uniform('Chigh2'   ,100  ,4000 ))])

Cmu      =  (np.log(10**3) + np.log(10**-1))/3                          
Csigma   =  (np.log(10**3) - np.log(10**-1))/3                          
C3       =   hp.lognormal('C3',       Cmu, Csigma) 

C4       =   hp.pchoice('C4',
       [(.75, hp.uniform('Csmall4'  ,0.001 ,5   )),
        (.20, hp.uniform('Cmed4'   ,5     ,100   )),
        (.05, hp.uniform('Chigh4'   ,100   ,4000   ))])

C5       =   hp.pchoice('C5',
      [(.75, hp.uniform('Csmall5'  ,0.1  ,5    )),
       (.25, hp.uniform('Cmed5'    ,5    ,100   ))])

C6       =   hp.pchoice('C6',
      [(.10, hp.uniform('Ctiny6'  ,0.001 ,0.1  )),
       (.65, hp.uniform('Csmall6' ,0.1   ,5    )),
       (.20, hp.uniform('Cmed6'   ,5     ,100  )),
       (.05, hp.uniform('Chigh6'  ,100   ,4000 ))])

# nsvc hyperparameters
nu       =   hp.uniform('nu'       ,0.1,  0.35 ) # nsvc; default=0.5
# nu     =   hp.uniform('nu'       ,0.1,  0.3  ) # nsvc; default=0.5


# plogrE1, plogrE2 hyperparameters
#    alpha or the elastic penalization ratio between the L1 and L2 penalty in the range [0,1]
low_ratio=   hp.uniform('low_ratio' ,0.01, 0.5  ) # plogrE1
high_ratio=  hp.uniform('high_ratio',0.5 , 0.99 ) # plogrE2


# dt, rf, gb, xgb hyperparameters
#    various tree and forest parameters
minleaf  =   hp.choice(   'minleaf', np.arange(1, 6, 1, dtype=int)) # dt, rf, gb, xgb; default=1; 1 to 5
minsplit = 2*minleaf                                                    # dt, rf, gb, xgb; default=2;
depth    =   hp.choice(     'depth', np.arange(2, 5, 1, dtype=int)) # gb, xgb;         default=3; 2 to 4
lrgb     =   hp.uniform(     'lrgb',     0.01,  3                 ) # gb;              default=0.1?
lrxgb    =   hp.uniform(     'lrxgb',    0.01,  1                 ) # xgb;             default=1
# old...
#lrgb    =   hp.uniform(     'lrgb',     0.01,  0.3               ) # gb, xgb;         
#minleaf =   hp.quniform( 'minleaf',        1,    5,  1           ) # dt, rf, gb, xgb; default=1; 1 to 5
#depth   =   hp.quniform(   'depth',        2,    4,  1           ) # gb, xgb;         default=3; 2 to 4

# for tabpfn Number of ensembles
en       =   hp.choice('en', (2, 3, 4, 5))

# svc, nsvc kernel hyperparameters
a        =   hp.pchoice(    'svc_a',                                # OISVgc;       slope of central region
      [(0.25,hp.choice(    'svc_a1',(1, 1.5, 2))),
       (0.75,hp.uniform(   'svc_ax', 0.7, np.exp(1)) )])

b        =   hp.pchoice(    'svc_b',                                # OISVgc, MSig; width
      [(0.25,hp.choice(  'svc_b0.5',(0.1, 0.5, 1, 2))),
       (0.75,hp.uniform(   'svc_bx', 0.05, 1) )])                       

b2       =   hp.uniform(   'svc_b2',       0.1,  0.75             ) # OISVgc; width

b3       =   hp.pchoice(   'svc_b3',                                # OISVgc, MSig; width
      [(0.25,hp.choice(   'svc_b3a',(0.001, 0.01, 0.1))),
       (0.75,hp.uniform(  'svc_b3b', 0.001, 100) )])                       

c        =   hp.pchoice(    'svc_c',                                # OISVgc;       bias to neg or pos match
      [(0.25,hp.choice(    'svc_c0',(-1, -0.5, 0, 0.5, 1))),
       (0.75,hp.uniform(   'svc_cx', -1,    1) )])                       

d        =   hp.pchoice(    'svc_d',                                # OISVgc, MSig; horizontal bias
      [(0.25,hp.choice(    'svc_d0',(-0.2, 0, 0.2))),
       (0.75,hp.uniform(   'svc_dx', -0.5, 0.5)    )])                       

sig      =   hp.uniform(    'svc_s', 10**( -2),  10**3            ) # RBF;          width
gamma    =   hp.uniform(        'g', 10**( -2),  10**3            ) # RBF;          width
#gamma   =   1/(2*sig**2)
#gamma   =   1/(2*b3**2)
siga     =   hp.uniform(   'svc_sa', 10**(-15),  20               ) # Sig;          slope
sigb     =   hp.uniform(   'svc_sb', 10**( -1),  10**1            ) #      SigN;    width
sigr     =   hp.uniform(   'svc_sr',        -5, -10**(-15)        ) # Sig, SigN;    vertical bias (intercept)
# old...
# a      =   hp.uniform(    'svc_a',         1,  np.exp(1)        ) # OISVgc;       slope of central region
# b      =   hp.uniform(    'svc_b',       0.1,  0.75             ) # OISVgc, MSig; width
# b      =   hp.uniform(    'svc_b',      0.05,  0.8              ) # OISVgc, MSig; width
# c      =   hp.uniform(    'svc_c',        -1,  1                ) # OISVgc;       bias to neg or pos match
# d      =   hp.normal(     'svc_d',         0,  0.2              ) # OISVgc, MSig; horizontal bias
# d      =   hp.normal(     'svc_d',      -0.5,  0.5              ) # OISVgc, MSig; horizontal bias
# d      =   hp.pchoice(    'svc_d',                                # OISVgc, MSig; horizontal bias
#     [(0.25,hp.choice(    'svc_d0',(-0.2, 0, 0.2))),
#      (0.75,hp.uniform(   'svc_dx', -1.5, 1.5) )])                       
# sig    =   hp.uniform(    'svc_s', 10**( -4),  10**6            ) #               legacy values


# nn hyperparameters
nodes    =   hp.choice(    'nodesx', ( 5, 10,    50,  100,  200, 300)) # default=100 
nn_alpha =   hp.choice( 'nn_alphax', (0.0001, 0.001, 0.01, 0.02     )) # default=0.0001
nn_batch =   hp.choice( 'nn_batchx', (   200,   300,  400,  500     )) # default=200
nn_batch0=   hp.choice( 'nn_batch0x', (   10,    30,   50,  100     )) # default=200
nn_i     =   hp.choice(     'nn_ix', (   200,   300                 )) # default=200
nn_tol   =   hp.choice(   'nn_tolx', (0.0001, 0.001                 )) # default=0.0001

