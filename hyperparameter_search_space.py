# hyperparameter_search_space.py
# Copyright 2020 André Carrington, Ottawa Hospital Research Institute
# Use is subject to the Apache 2.0 License
# Written by André Carrington
#
#    cross validation functions
from sklearn.model_selection import cross_val_score  # only permits 1 score
from sklearn.model_selection import cross_validate   # permits multiple scores, for future use
from xgboost                 import cv     as xgbcv
#from catboost               import cv     as cbcv

# a global value used in most classifier hyperparameter settings
#opt_measure = 'precision_macro'   # main measure for optimization
#main_cv  = cross_validate      # main function for cross validation, permits multiple scores
#main_cv  = 'cross_val_score'   # main function for cross validation, permits only 1 score
#xgb_cv   = 'xgbcv'
opt_measure  = 'roc_auc'           # main measure for optimization, score to use in cross validation
main_cv      = cross_val_score     # main function for cross validation, permits only 1 score
xgb_cv       = xgbcv

# knn,  defaults: metric='minkowski', p=2, algorithm='auto', leaf_size=30 
#       changes : n_neighbors=k, weights='distance'
space_knn       = {'n_neighbors':k}
dictx_knn       = dict(classifier=knn, name='knn', space=space_knn, fixed=[],
                       measure=opt_measure, fixed2=[], cv=main_cv)
cl = addcl(cl, dictx_knn, {})


# nb,   defaults: priors, var_smoothing (...these are not typically changed)
space_nb        = [] #
dictx_nb        = dict(classifier=nb, name='nb', space=space_nb, fixed=[],
                       measure=opt_measure, fixed2=[], cv=main_cv)
cl = addcl(cl, dictx_nb, {})

          
# logr, defaults: penalty='l2', dual=False, fit_intercept=True
#       changes : max_iter=100, C=C, class_weight='balanced' (rebalanced)
dictx_logr      = dict(classifier=logr, measure=opt_measure, fixed2=[], 
                       cv=main_cv)

fixed_logr      = {'penalty':'none'      ,'class_weight':'balanced','solver':'sag'}
#fixed_logr      = {'penalty':'none'      ,'class_weight':'balanced','solver':'lbfgs'}
space_logr      = {}
#space_logr     = []
dictx_logrB     = dict(name  ='logr'  , space=space_logr, fixed=fixed_logr)
cl = addcl(cl, dictx_logr, dictx_logrB)

space_plogr     = {'C':C}
fixed_plogr1    = {'penalty':'l1'        ,'class_weight':'balanced','solver':'liblinear'}
#fixed_plogr1    = {'penalty':'l1'        ,'class_weight':'balanced','solver':'saga'}
dictx_plogr1    = dict(name  ='plogr1', space=space_plogr, fixed=fixed_plogr1)
cl = addcl(cl, dictx_logr, dictx_plogr1)

#space_plogr as before
fixed_plogr2    = {'penalty':'l2'        ,'class_weight':'balanced'}
dictx_plogr2    = dict(name  ='plogr2', space=space_plogr, fixed=fixed_plogr2)
cl = addcl(cl, dictx_logr, dictx_plogr2)

fixed_plogrE    = {'penalty':'elasticnet','class_weight':'balanced','solver':'saga'}
space_plogrE1   = {'C':C,'l1_ratio':low_ratio}
dictx_plogrE1   = dict(name  ='plogrE1', space=space_plogrE1, fixed=fixed_plogrE)
cl = addcl(cl, dictx_logr, dictx_plogrE1)

#fixed_plogrE as before
space_plogrE2   = {'C':C,'l1_ratio':high_ratio}
dictx_plogrE2   = dict(name  ='plogrE2', space=space_plogrE2, fixed=fixed_plogrE)
cl = addcl(cl, dictx_logr, dictx_plogrE2)


# dt,   defaults: criterion='gini', splitter='best', max_depth=None,  
#                  max_features='sqrt', max_leaf_nodes=None, ccp_alpha=0.0
#       changes:  class_weight='balanced', min_samples_leaf=1, min_samples_split=2
fixed_dt = {'class_weight':'balanced'}
space_dt = {'min_samples_leaf':minleaf, 'min_samples_split':minsplit }
dictx_dt = dict(classifier=dt, name='dt', space=space_dt, fixed=fixed_dt,
                measure=opt_measure, fixed2=[], cv='cross_val_score')
cl = addcl(cl, dictx_dt, {})


# rf,   as rf100 and rf200 re n_estimators
#       defaults: bootstrap=True, max_depth=None,
#                 max_features='sqrt', max_leaf_nodes=None, ccp_alpha=0.0,      
#       changes:  n_estimators=100, class_weight='balanced', min_samples_leaf=1, 
#                 min_samples_split=2
fixed_rf     = {'class_weight':'balanced'}
space_rf     = {'min_samples_leaf':minleaf, 'min_samples_split':minsplit }

fixed_rf_0   = {'n_estimators':20}
dictx_rf_0   = dict(classifier=rf, name='rf0', space=space_rf, fixed=fixed_rf,
                    measure=opt_measure, fixed2=fixed_rf_0, cv=main_cv)
cl = addcl(cl, dictx_rf_0, {})

fixed_rf_100 = {'n_estimators':100}
dictx_rf_100 = dict(classifier=rf, name='rf1', space=space_rf, fixed=fixed_rf,
                    measure=opt_measure, fixed2=fixed_rf_100, cv=main_cv)
cl = addcl(cl, dictx_rf_100, {})

#fixed_rf as before
#space_rf as before
fixed_rf_200 = {'n_estimators':200}
dictx_rf_200 = dict(classifier=rf, name='rf2', space=space_rf, fixed=fixed_rf,
                    measure=opt_measure, fixed2=fixed_rf_200, cv=main_cv)
cl = addcl(cl, dictx_rf_200, {})



# gb,   as gb100 and gb200 re n_estimators
#   loss as deviance required for calibrated scores (predict_proba)
#       defaults: loss='deviance', subsample=1.0, criterion='friedman_mse', 
#                 max_features=None, max_leaf_nodes=None, warm_start=False, 
#                 validation_fraction=0.1, n_iter_no_change=None, tol=0.0001,
#                 ccp_alpha=0.0
#       changes:  n_estimators=100, learning_rate=0.1, max_depth=depth, min_samples_leaf=1,
#                 min_samples_split=2
fixed_gb      = {'subsample':1.0}
space_gb      = {'learning_rate':lrgb, 'max_depth':depth, 'min_samples_leaf':minleaf, 
                 'min_samples_split':minsplit }

fixed_gb_100  = {'n_estimators':100}
dictx_gb_100  = dict(classifier=gb, name='gb1', space=space_gb, fixed=fixed_gb,
                     measure=opt_measure, fixed2=fixed_gb_100, cv=main_cv)
cl = addcl(cl, dictx_gb_100, {})

#fixed_gb as before
#space_gb as before
fixed_gb_200  = {'n_estimators':200}
dictx_gb_200  = dict(classifier=gb, name='gb2', space=space_gb, fixed=fixed_gb,
                     measure=opt_measure, fixed2=fixed_gb_200, cv=main_cv)
cl = addcl(cl, dictx_gb_200, {})

fixed_gb_300  = {'n_estimators':300}
dictx_gb_300  = dict(classifier=gb, name='gb3', space=space_gb, fixed=fixed_gb,
                     measure=opt_measure, fixed2=fixed_gb_300, cv=main_cv)
cl = addcl(cl, dictx_gb_300, {})

fixed_gb_400  = {'n_estimators':400}
dictx_gb_400  = dict(classifier=gb, name='gb4', space=space_gb, fixed=fixed_gb,
                     measure=opt_measure, fixed2=fixed_gb_400, cv=main_cv)
cl = addcl(cl, dictx_gb_400, {})


# lsvc, uses libSVM (but linear SVC is fast anyway)
#       defaults: penalty='l2', loss='squared_hinge', tol=0.0001, C=1.0, 
#                 multi_class='ovr', fit_intercept=True, intercept_scaling=1, max_iter=1000
#       changes:  C=C, class_weight='balanced', dual=lsvc_dual
#  For the linearSVC classifier follow sklearn promoted best practice to
#  use the primal formula for "thin" clinical data where instances > features, 
#  otherwise use the dual formula; note: the dual is necessary/standard for non-linear kernels
space_lsvc    = {'C':C5}
fixed_lsvc_t  = {'class_weight':'balanced', 'dual':False}
dictx_lsvc_t  = dict(classifier=lsvc, name='lsvc_t', space=space_lsvc, fixed=fixed_lsvc_t,
                     measure=opt_measure, fixed2=[], cv=main_cv)
cl = addcl(cl, dictx_lsvc_t, {})

# for "wide" genomic data, use the dual form of the objective function
#space_lsvc as before
fixed_lsvc_w  = {'class_weight':'balanced', 'dual':True}
dictx_lsvc_w  = dict(classifier=lsvc, name='lsvc_w', space=space_lsvc, fixed=fixed_lsvc_w,
                     measure=opt_measure, fixed2=[], cv=main_cv)
cl = addcl(cl, dictx_lsvc_w, {})


             
# svc,  uses libSVM and therefore Sequential Minimal Optimization (SMO) which is 10x 
#       faster than Quadratic Programming (QP) with a sometimes tiny loss in accuracy
#       (QP is perfectly optimal)
#
#       defaults:  tol=0.001, max_iter=-1
#       immutable: (penalty='l2', loss='squared_hinge')
#       changes:   C=C, kernel='rbf', gamma='scale' or float, coef0=0.0, shrinking=False, 
#                  cache_size=200, class_weight='balanced', probability=True
#       note:     probability=True involves a 2nd internal cross-validation loop to fit
#                 probabilities, so this can add a lot of time/cpu cycles
#       note:     do not use class_weight='auto', internet discussions indicate lack
#                 or proper rationale/formula for it
#       note:     do not use class_weight='balanced' unless you are sure that the version
#                 of sklearn you have supports it.
#
# space_svc_RBF = dict(C=C, gamma=gamma) # alternate format
# inverse_weight= np.array([P/(P+N), N/(P+N)]) # alternate for class_weight
# measure       = 'precision_macro'
#
fixed_svc       = {'coef0':0.0, 'shrinking':False, 'cache_size':3000, 
                   'class_weight':'balanced', 'probability':True}
#fixed_svc       = {'coef0':0.0, 'shrinking':True, 'cache_size':3000, 
#                   'class_weight':'balanced', 'probability':True}
# fixed_svc     = {'coef0':0.0, 'shrinking':False, 'cache_size':200, 
#                  'class_weight':'balanced', 'probability':False}

dictx_svc       = dict(classifier=svc, measure=opt_measure, fixed=fixed_svc, cv=main_cv)

space_svc_Pwr1  = {'C':C5                                           }
fixed_svc_Pwr1 = {'kernel':Pwr1_kernel, gamma:1}
dictx_svc_Pwr1  = dict(name='svcPwr1'   , space=space_svc_Pwr1  , fixed2=fixed_svc_Pwr1)
cl = addcl(cl, dictx_svc, dictx_svc_Pwr1)
#
space_svc_Log1  = {'C':C5                                           }
fixed_svc_Log1  = {'kernel':Log1_kernel, gamma:1}
dictx_svc_Log1  = dict(name='svcLog1'   , space=space_svc_Log1  , fixed2=fixed_svc_Log1)
cl = addcl(cl, dictx_svc, dictx_svc_Log1)
#
space_svc_Lin   = {'C':C5                                           }
fixed_svc_Lin   = {'kernel':'linear', 'gamma':1}
dictx_svc_Lin   = dict(name='svcLin'   , space=space_svc_Lin   , fixed2=fixed_svc_Lin)
cl = addcl(cl, dictx_svc, dictx_svc_Lin)
#
space_svc_RBF   = {'C':C5,           'gamma':gamma                  }
fixed_svc_RBF   = {'kernel':'rbf'}
dictx_svc_RBF   = dict(name='svcRBF'   , space=space_svc_RBF   , fixed2=fixed_svc_RBF)
cl = addcl(cl, dictx_svc, dictx_svc_RBF)
#
space_svc_Sig   = {'C':C5,'gamma':siga,                  'coef0':sigr}
fixed_svc_Sig   = {'kernel':'sigmoid'}
dictx_svc_Sig   = dict(name='svcSig'   , space=space_svc_Sig   , fixed2=fixed_svc_Sig)
cl = addcl(cl, dictx_svc, dictx_svc_Sig)
#
space_svc_SigN  = {'C':C5,    'a':sigb,                      'r':sigr}
fixed_svc_SigN  = {'kernel':SigN_kernel, 'gamma':1}   # custom kernel hyperparms in space (except C for svc)
dictx_svc_SigN  = dict(name='svcSigN'  , space=space_svc_SigN  , fixed2=fixed_svc_SigN)
cl = addcl(cl, dictx_svc, dictx_svc_SigN)
#
space_svc_MSig  = {'C':C5,               'b':b    ,          'd':d   }
fixed_svc_MSig  = {'kernel':MSig_kernel, 'gamma':1}   # custom kernel hyperparms in space (except C for svc)
dictx_svc_MSig  = dict(name='svcMSig'  , space=space_svc_MSig  , fixed2=fixed_svc_MSig)
cl = addcl(cl, dictx_svc, dictx_svc_MSig)
#
space_svc_MSig0  = {'C':C5,               'b':b }
fixed_svc_MSig0  = {'kernel':MSig0_kernel, 'gamma':1}   # custom kernel hyperparms in space (except C for svc)
dictx_svc_MSig0  = dict(name='svcMSig0'  , space=space_svc_MSig0  , fixed2=fixed_svc_MSig0)
cl = addcl(cl, dictx_svc, dictx_svc_MSig0)
#
space_svc_OISVgc= {'C':C5,    'a':a   ,  'b':b2   , 'c':c,   'd':d   }
fixed_svc_OISVgc= {'kernel':OISVgc_kernel, 'gamma':1} # custom kernel hyperparms in space (except C for svc)
dictx_svc_OISVgc= dict(name='svcOISVgc', space=space_svc_OISVgc, fixed2=fixed_svc_OISVgc)
cl = addcl(cl, dictx_svc, dictx_svc_OISVgc)
#
#space_svc_OISVneg= {'C':C5,    'a':a   ,  'b':b2   , 'd':d   }
space_svc_OISVneg= {'C':C5,    'b':b   }
fixed_svc_OISVneg= {'kernel':OISVneg_kernel, 'gamma':1} # custom kernel hyperparms in space (except C for svc)
dictx_svc_OISVneg= dict(name='svcOISVneg', space=space_svc_OISVneg, fixed2=fixed_svc_OISVneg)
cl = addcl(cl, dictx_svc, dictx_svc_OISVneg)
#
space_svc_OISVpos= {'C':C5,    'b':b   }
fixed_svc_OISVpos= {'kernel':OISVpos_kernel, 'gamma':1} # custom kernel hyperparms in space (except C for svc)
dictx_svc_OISVpos= dict(name='svcOISVpos', space=space_svc_OISVpos, fixed2=fixed_svc_OISVpos)
cl = addcl(cl, dictx_svc, dictx_svc_OISVpos)
#
space_svc_OSig  = {'C':C5,               'b':b    , 'c':c            }
space_svc_OSig  = {'C':C5,               'b':b    , 'c':c            }
fixed_svc_OSig  = {'kernel':OSig_kernel, 'gamma':1}   # custom kernel hyperparms in space (except C for svc)
dictx_svc_OSig  = dict(name='svcOSig' , space=space_svc_OSig , fixed2=fixed_svc_OSig)
cl = addcl(cl, dictx_svc, dictx_svc_OSig)
    

    
# nsvc, defaults: degree=3, tol=0.001, cache_size=200, max_iter=-1, 
#                 decision_function_shape='ovr', break_ties=False, random_state=None
#       immutable: (penalty='l2', loss='squared_hinge')
#       changes:  nu=0.5, kernel='rbf', gamma='scale', coef0=0.0, shrinking=False, 
#                 cache_size=200, class_weight='balanced', probability=True,
#       note:     gamma applies to rbf and sigmoid
fixed_nsvc       = {'coef0':0.0, 'shrinking':False, 'cache_size':200, 
                    'class_weight':'balanced', 'probability':True}

dictx_nsvc       = dict(classifier=nsvc, measure=opt_measure, fixed=fixed_nsvc, cv=main_cv)

space_nsvc_RBF   = {'nu':nu,         'gamma':gamma                   }
fixed_nsvc_RBF   = {'kernel':'rbf'}         # custom kernel hyperparms in space (except nu for nsvc)
dictx_nsvc_RBF   = dict(name='nsvcRBF'   , space=space_nsvc_RBF   , fixed2=fixed_nsvc_RBF)
cl = addcl(cl, dictx_nsvc, dictx_nsvc_RBF)  
#
space_nsvc_OISVgc= {'nu':nu,   'a':a  ,  'b':b    , 'c':c,   'd':d   }
fixed_nsvc_OISVgc= {'kernel':OISVgc_kernel} # custom kernel hyperparms in space (except nu for nsvc)
dictx_nsvc_OISVgc= dict(name='nsvcOISVgc', space=space_nsvc_OISVgc, fixed2=fixed_nsvc_OISVgc)
cl = addcl(cl, dictx_nsvc, dictx_nsvc_OISVgc)



# xgb,  defaults: booster='gbtree', min_child_weight=1, gamma=0, 
#                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
#                 colsample_bynode=?
#       changes:  objective='binary:logistic', learning_rate=0.01-3, max_depth=3-10, 
#                 reg_lambda=1, reg_alpha=0, scale_pos_weight=N/P, 
#                 tree_method=hist (or gpu_hist)
#
#                 note: tree_method=auto chooses between exact (small data) or approx 
#                       (big data) but hist for big data is faster and gpu_hist even better
fixed_xgb     = {'objective':'binary:logistic','reg_lambda':1,'reg_alpha':0, 
                 'scale_pos_weight':N/P, 'tree_method':'hist'}
space_xgb     = {'learning_rate':lrxgb, 'max_depth':depth, 'min_samples_leaf':minleaf, 
                 'min_samples_split':minsplit }
fixed_xgb_100 = {'n_estimators':100}
dictx_xgb_100 = dict(name='xgb1', classifier=xgb, space=space_xgb, fixed=fixed_xgb,
                     measure='map', fixed2=fixed_xgb_100, cv=xgb_cv)
#dictx_xgb_100 = dict(name='xgb1', classifier=xgb, space=space_xgb, fixed=fixed_xgb,
#                     measure='map', fixed2=fixed_xgb_100, cv=xgb_cv)
cl = addcl(cl, dictx_xgb_100, {})

#fixed_xgb as above
#space_xgb as above
fixed_xgb_200 = {'n_estimators':200}
dictx_xgb_200 = dict(name='xgb2', classifier=xgb, space=space_xgb, fixed=fixed_xgb,
                     measure='map', fixed2=fixed_xgb_200, cv=xgb_cv)
cl = addcl(cl, dictx_xgb_200, {})

fixed_xgb_300 = {'n_estimators':300}
dictx_xgb_300 = dict(name='xgb3', classifier=xgb, space=space_xgb, fixed=fixed_xgb,
                     measure='map', fixed2=fixed_xgb_300, cv=xgb_cv)
cl = addcl(cl, dictx_xgb_300, {})


# tpfn, defaults: device='cpu', N_ensemble_configurations=3, combine_preprocessing=False,
#                 no_preprocess_mode=False, multiclass_decoder='permutation',
#                 feature_shift_decoder=True, only_inference=True, seed=0
#       changes:  N_ensemble_configurations
fixed_tpfn    = {'device':'cpu', 'combine_preprocessing':False, 'no_preprocess_mode':False,
                 'multiclass_decoder':'permutation', 'feature_shift_decoder':True,
                 'only_inference':True, 'seed':0, 'N_ensemble_configurations':3}
space_tpfn    = []
dictx_tpfn    = dict(name='tpfn', classifier=tpfn, space=space_tpfn, fixed=fixed_tpfn,
                     measure=opt_measure, fixed2=[], cv=main_cv)
cl = addcl(cl, dictx_tpfn, {})


# cb catboost
fixed_cb  = {'cat_features':cat_features, 'silent':True}
space_cb  = {'learning_rate':lrxgb}
dictx_cb  = dict(name='cb', classifier=cb, space=space_cb, fixed=fixed_cb,
                 measure=opt_measure, fixed2=[], cv=main_cv)
cl = addcl(cl, dictx_cb, {})
#fixed_cb  = {'objective':'Logloss', #'reg_lambda':1, # AUC, HingeLoss, BrierScore
#             'scale_pos_weight':N/P}
#space_cb  = {'learning_rate':lrgb, 'max_depth':depth, 'min_data_in_leaf':minleaf }
#dictx_cb  = dict(name='cb', classifier=cb, space=space_cb, fixed=fixed_cb,
#                 measure='Logloss', fixed2=[], cv=cbcv)



# nn,   notes:    no nadam solver
#       defaults: activation='relu', solver='adam', shuffle=True, validation_fraction=0.1,
#                 n_iter_no_change=10
#       changes:  hidden_layer_sizes=(100, ), alpha=0.0001, batch_size=200,
#                 max_iter=200, tol=0.0001,  warm_start=False, 
#                 random_state=rngSeed,  early_stopping=True
#       for_adam: beta_1=0.9, beta_2=0.999, epsilon=1e-08
#       for_sgd:  learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
#                 momentum=0.9, nesterovs_momentum=True
#       for_lbfgs: max_fun=15000
fixed_nn       = {'warm_start':False,'random_state':1,'early_stopping':True}
fixed_nn_relu  = {'activation':'relu'}
space_nn0      = {'hidden_layer_sizes':(nodes, ),'alpha':nn_alpha,'batch_size':nn_batch0,
                  'max_iter':nn_i,'tol':nn_tol}
dictx_nn0      = dict(classifier=nn, space=space_nn0, fixed=fixed_nn, measure=opt_measure,
                      cv=main_cv)
dictx_nn_relu  = dict(name='nnReLU0', fixed2=fixed_nn_relu)
cl = addcl(cl, dictx_nn0, dictx_nn_relu)

space_nn       = {'hidden_layer_sizes':(nodes, ),'alpha':nn_alpha,'batch_size':nn_batch,
                  'max_iter':nn_i,'tol':nn_tol}
dictx_nn       = dict(classifier=nn, space=space_nn, fixed=fixed_nn, measure=opt_measure, 
                      cv=main_cv)
dictx_nn_relu  = dict(name='nnReLU', fixed2=fixed_nn_relu)
cl = addcl(cl, dictx_nn, dictx_nn_relu)  

#dictx_nn as above
fixed_nn_tanh  = {'activation':'tanh'}
dictx_nn_tanh  = dict(name='nnTanh', fixed2=fixed_nn_tanh)
cl = addcl(cl, dictx_nn, dictx_nn_tanh)  
