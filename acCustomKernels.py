# acCustomKernels.py
# Copyright 2020 André Carrington, Ottawa Hospital Research Institute
# Use is subject to the Apache 2.0 License
# Written by André Carrington
#
# Specialized kernels for kernel methods like SVM/SVC, SVR
# Made primarily for classification. In the future, some of these kernels
# will be fully-parameterized for each feature dimension individually.
#
# functions:
#
#    double_sigmoid_g
#    double_sigmoid_h
#
#     OISVg_function
#     OISVh_function
#    OISVgc_function
#    OISVhc_function
#      ISVg_function
#      ISVh_function
#
#      OSig_kernel
#    OISVgc_kernel
#    OISVhc_kernel
#      MSig_kernel
#      SigN_kernel
#
#      ISVg_kernel
#      ISVh_kernel
#   
#     OMIcs_kernel (tbc)
#     OMIcp_kernel (tbc)
#     
# tbc = to be completed from existing Matlab code and dissertation (2018)
import numpy as np

def double_sigmoid_g(b, d):   
    # Carrington (2018) created the world's first (to our knowledge) double-sigmoid or double
    # S curve (in two different ways) that is continuous in the first and second derivatives 
    # (and so on). Previous definitions are piece-wise defined. This is a special case of the
    # OISVg function (which follows). The "other" way pertains to OISVh.
    #
    # b = width
    xs  = x/b         # x scaled by b
    return -xs*np.exp(-xs**2) + np.tanh(xs)
    #
    # the following is equivalent:
    #    a=1 flat central plateau
    #    c=0 no vertical   shift 
    #    d=0 no horizontal shift
    # return OISVg_function(x, a=1, b=b, c=0, d=0)
#enddef

def double_sigmoid_h(b, d):   
    # Carrington (2018) created the world's first (to our knowledge) double-sigmoid or double
    # S curve (in two different ways) that is continuous in the first and second derivatives 
    # (and so on). Previous definitions are piece-wise defined. This is a special case of the
    # OISVh function (which follows). The "other" way pertains to OISVg.
    #
    # b = width
    xs  = x/b         # x scaled by b
    return np.tanh(xs)*(1 - np.sech(xs)**2)
    #
    # the following is equivalent:
    #    a=1 flat central plateau
    #    c=0 no vertical   shift 
    #    d=0 no horizontal shift
    # return OISVh_function(x, a=1, b=b, c=0, d=0)
#enddef

def ISVg_function(x, a, b, d):
    # Carrington's "Insensitive Sigmoid Variant" (ISV) basis or transfer function (2018),
    # vectorized, using a Gaussian (g)
    #
    # Is a sigmoid function with a flat or sloped plateau of insensitivity in the middle
    #
    # a = central plateau: a=1 flat, a>1 positive slope, a<1 negative slope (not recommended)
    #                      a>exp(1) no plateau/inflection (becomes a single S curve)
    # b = width
    # d = horizontal shift
    xss   = (x-d)/b     # x horizontally shifted by d and scaled by b
    return -(xss/a)*np.exp(-xss**2) + np.tanh(xss)
    #
    # the following is equivalent:
    #    c=0 no vertical shift (in a kernel this makes the positive/negative match weight equal/symmetric)
    # return OISVg_function(x, a=a, b=b, c=0, d=d)
#enddef

def ISVh_function(x, a, b, d):
    # Carrington's "Insensitive Sigmoid Variant" (ISV) basis or transfer function (2018),
    # vectorized, using a hyperbolic tangent (h)
    #
    # Is a sigmoid function with a flat or sloped plateau of insensitivity in the middle.
    #
    # a = central plateau: a=1 flat, a>1 positive slope, a<1 negative slope (not recommended)
    #                      a>exp(1) no plateau/inflection (becomes a single S curve)
    # b = width
    # d = horizontal shift
    xss   = (x-d)/b     # x horizontally shifted by d and scaled by b
    return np.tanh(xss)*(1 - (1/a)*np.sech(xss)**2)
    #
    # the following is equivalent:
    #    c=0 no vertical shift (in a kernel this makes the positive/negative match weight equal/symmetric)
    # return OISVh_function(x, a=a, b=b, c=0, d=d)
#enddef

def OISVg_function(x, a, b, c, d):
    # Carrington's "Orthant Insensitive Sigmoid Variant" (OISV) basis or transfer function (2018),
    # vectorized, using a Gaussian (g)
    #
    # Is a generalization of a double-sigmoid (or double S curve) that has a plateau in the middle.
    # As a generalization it can have a wider or narrower plateau, or a plateau with a slope 
    # ranging from slight, to great, to opposite.  Morphs between the regular S curve and double S
    # curve and generalizes both.
    #
    # a = central plateau: a=1 flat, a>1 positive slope, a<1 negative slope (not recommended)
    #                      a>exp(1) no plateau/inflection (becomes a single S curve)
    # b = width
    # c = vertical shift (in a kernel this creates positive/negative match asymmetry)
    # d = horizontal shift
    #
    xss   = (x-d)/b                  # x horizontally shifted by d and scaled by b
    return -(xss/a)*np.exp(-xss**2) + np.tanh(xss) + c    
#enddef

def OISVh_function(x, a, b, c, d):
    # Carrington's "Orthant Insensitive Sigmoid Variant" (OISV) basis or transfer function (2018),
    # vectorized, using a hyperbolic tangent (h)
    #
    # Is a generalization of a double-sigmoid (or double S curve) that has a plateau in the middle.
    # As a generalization it can have a wider or narrower plateau, or a plateau with a slope 
    # ranging from slight, to great, to opposite.  Morphs between the regular S curve and double S
    # curve and generalizes both.
    #
    # a = central plateau: a=1 flat, a>1 positive slope, a<1 negative slope (not recommended)
    #                      a>exp(1) no plateau/inflection (becomes a single S curve)
    # b = width
    # c = vertical shift (in a kernel this creates positive/negative match asymmetry)
    # d = horizontal shift
    #
    xss   = (x-d)/b                  # x horizontally shifted by d and scaled by b
    return np.tanh(xss)*(1 - (1/a)*np.sech(xss)**2) + c        
#enddef

def OISVgc_function(x, a, b, c, d):    
    # Carrington's "Orthant Insensitive Sigmoid Variant" (OISV) basis or transfer function (2018),
    # vectorized, using a Gaussian with constraints (gc)
    #
    # See the description of the OISVg function.
    #
    # The constraints force parameters b and d to have similar values and effects as the same 
    # parameters in Carrington's Mercer Sigmoid and Orthant Sigmoid kernels for ease of
    # interpretation, implementation and comparison
    #
    # a = central plateau: a=1 flat, a>1 positive slope, a<1 negative slope (not recommended)
    #                      a>exp(1) no plateau/inflection (becomes a single S curve)
    # b = width
    # c = vertical shift (in a kernel this creates positive/negative match asymmetry)
    # d = horizontal shift
    #
    d2    = d*np.abs(np.tanh( (2/0.375)*(b-0.25-0.375) ))
    xss   = (x-d2)/b                  # x horizontally shifted by d2 and scaled by b
    return -(xss/a)*np.exp(-(xss**2)) + np.tanh(xss) + c
    # return OISVg_function(x, a=a, b=b, c=c, d=d2)
#enddef

def OISVhc_function(x, a ,b, c, d):
    # Carrington's "Orthant Insensitive Sigmoid Variant" (OISV) basis or transfer function (2018),
    # vectorized, using a hyperbolic tangent with constraints (hc)
    #
    # See the description of the OISVh function.
    #
    # The constraints force parameters b and d to have similar values and effects as the same 
    # parameters in Carrington's Mercer Sigmoid and Orthant Sigmoid kernels for ease of
    # interpretation, implementation and comparison
    #  
    # a = central plateau: a=1 flat, a>1 positive slope, a<1 negative slope (not recommended)
    #                      a>exp(1) no plateau/inflection (becomes a single S curve)
    # b = width
    # c = vertical shift (in a kernel this creates positive/negative match asymmetry)
    # d = horizontal shift
    #
    d2    = d*np.abs(np.tanh( (2/0.375)*(b-0.25-0.375) ))
    xss   = (x-d2)/b                  # x horizontally shifted by d and scaled by b
    return np.tanh(xss)*(1 - (1/a)*np.sech(xss)**2) + c
    # return OISVh_function(x, a=a, b=b, c=c, d=d2)
#enddef

def OSig_kernel(b=0.25, c=1, d=0): # wrapper for Carrington's OSig kernel
    # Returns a 2 input callable kernel for kernel methods like SVM/SVC, SVR
    #
    # Orthant Sigmoid (OSig) kernel - a kernel designed for classification that gives
    #   greater weight toward the all-positive orthant (a quadrant in 2D) or the all-
    #   negative orthant.
    #       - for nominal data that are one-hot encoded as binary indicators where the 
    #         presence (positives) have much greater meaning than the absence (negatives)
    #       - for presence/absence data, such as:
    #            - the use of top descriptive keywords, as in skin lesions.  Top keywords are
    #              non-exhaustive.  Some keywords which might or should apply partially or fully
    #              are not indicated.  Their presence matters weights much more than absence, if
    #              absence is to be weighted at all.
    #            - observations of animals/organisms in ecology. Presence in the time and area
    #              of observation is informative, while absence of an observation is a lack of 
    #              information and may or may not indicate actual absence.  Hence, presence 
    #              matters weights much more than absence, if absence is to be weighted at all.
    #            - the aforementioned observations with probability or frequency weighting
    #            - imputed continuous values from binary data, indicating probability/uncertainty
    #       - has a similar purpose to the azzoo distance measure made to measure the distance
    #         between two binary vectors; except our kernel works with binary indicators,
    #         ordinals or continuous numbers.  A numeric (continuous number) kernel that can
    #         work effectively with non-numeric data types and allow for a rich set of uses and
    #         interpretations. 
    #
    # b = width
    # c = assymetric weight, c>0 weighs +ve more, c<0 vice-versa, c=1 weighs +ve only
    # d = horizontal shift
    #
    # Parameter defaults are suitable for standardization to 2 or 3 sigma
    
    def OSig_hidden(U, V, b, c, d):       # OSig kernel (Carrington, 2018)
    
        OSig_phi = lambda x, b, c, d: np.tanh((x-d)/b)+c # Orthant Sigmoid basis function, vectorized
                                      # a vertical shift in the basis function, adds weight to the
                                      # all positive orthant in a kernel for a positive shift 
                                      # (and vice-versa for a negative shift)

        p   = U.shape[1]              # p is number of feature dimensions
        ku  = OSig_phi(U.T,b,c,d)     # matrix as input and output to phi, with
        kv  = OSig_phi(V.T,b,c,d)     #   ku, kv, U', V' as vertical instance p-vectors
        K   = np.dot(ku.T,kv)         # use Genton's (2002) mechanism G=w*z' 
                                      #   where w =ku' is one or more horizontal instance vectors
                                      #   and   z'=kv  is one or more vertical   instance vectors    
                                      #   for explicit Mercer kernels (Carrington, 2018)
        K   = (K + p*(1-c**2)) / (2 * (np.abs(c)+1) )
        return K
    #enddef

    return lambda U, V: OSig_hidden(U, V, b, c, d)
#enddef

def OISVgc_kernel(a=1, b=0.5, c=1, d=0):
    # Returns a 2 input callable kernel for kernel methods like SVM/SVC, SVR
    #
    # This kernel has the same behaviour, and almost the same shape, as the OISVhc kernel.
    #
    # Carrington's Orthant Insensitive Sigmoid Variant (OISV) kernel using a Gaussian with 
    # constraints (gc) (2018). A kernel designed for classification that has a central region 
    # that is insensitive (i.e. flat and unweighted) which can ignore the effect data near the 
    # center and thus likely near the class boundary, where class noise/error is greatest.
    #
    # It has all the properties of the Orthant Sigmoid (OSig) kernel, as a generalization of it. 
    #
    # a = central plateau: a=1 flat, a>1 positive slope, a<1 negative slope (not recommended)
    #                      a>exp(1) no plateau/inflection (becomes a single S curve)
    # b = width
    # c = assymetric weight, c>0 weighs positive more, c<0 vice-versa, c=1 weighs positive only
    # d = horizontal shift
    #
    # Parameter defaults are suitable for standardization to 2 or 3 sigma
    # Can be further generalized to parameter values in each dimension.
    return lambda U, V: OISVgc_hidden(U, V, a, b, c, d)
#enddef

#def OISVneg_kernel(a=1, b=0.5, d=0):
#    # See OISVgc_kernel for description/explanation.
#    c = -1  # match negatives only
#    return lambda U, V: OISVgc_hidden(U, V, a, b, c, d)
##enddef

def OISVpos_kernel(b=0.5):
    # See OISVgc_kernel for description/explanation.
    a =  1
    c = +1  # match positives only
    d =  0
    return lambda U, V: OISVgc_hidden(U, V, a, b, c, d)
#enddef

def OISVneg_kernel(b=0.5):
    # See OISVgc_kernel for description/explanation.
    a =  1
    c = -1  # match negatives only
    d =  0
    return lambda U, V: OISVgc_hidden(U, V, a, b, c, d)
#enddef

def OISVgc_hidden(U, V, a, b, c, d):   # OISVgc kernel (Carrington, 2018)
    # See OISVgc_kernel for description/explanation.
    OISVgc_phi = OISVgc_function
    p   = U.shape[1]              # p is number of feature dimensions
    ku  = OISVgc_phi(U.T, a, b, c,d) # matrix as input and output to phi, with
    kv  = OISVgc_phi(V.T, a, b, c, d) #   ku, kv, U', V' as vertical instance p-vectors
    K   = np.dot(ku.T, kv)         # use Genton's (2002) mechanism G=w*z'
    #   where w =ku' is one or more horizontal instance vectors
    #   and   z'=kv  is one or more vertical   instance vectors
    #   for explicit Mercer kernels (Carrington, 2018)
    K   = (K + p*(1-c**2)) / (2 * (np.abs(c)+1) )
    return K
#enddef

def OISVhc_kernel(a=1, b=0.5, c=1, d=0):
    # Returns a 2 input callable kernel for kernel methods like SVM/SVC, SVR
    #
    # This kernel has the same behaviour, and almost the same shape, as the OISVgc kernel.
    #
    # Carrington's Orthant Insensitive Sigmoid Variant (OISV) kernel using a hyperbolic tangent 
    # with constraints (hc) (2018). A kernel designed for classification that has a central 
    # region that is insensitive (i.e. flat and unweighted) which can ignore the effect data near 
    # the center and thus likely near the class boundary, where class noise/error is greatest.
    #
    # It has all the properties of the Orthant Sigmoid (OSig) kernel, as a generalization of it. 
    #
    # a = central plateau: a=1 flat, a>1 positive slope, a<1 negative slope (not recommended)
    #                      a>exp(1) no plateau/inflection (becomes a single S curve)
    # b = width
    # c = assymetric weight, c>0 weighs positive more, c<0 vice-versa, c=1 weighs positive only
    # d = horizontal shift
    #
    # Parameter defaults are suitable for standardization to 2 or 3 sigma
    # Can be further generalized to parameter values in each dimension.
    
    def OISVhc_hidden(U, V, a, b, c, d):   # OISVhc kernel (Carrington, 2018)
        
        OISVhc_phi = OISVhc_function
        p   = U.shape[1]                   # p is number of feature dimensions
        ku  = OISVhc_phi(U.T, a, b, c, d)  # matrix as input and output to phi, with
        kv  = OISVhc_phi(V.T, a, b, c, d)  #   ku, kv, U', V' as vertical instance p-vectors
        K   = np.dot(ku.T, kv)             # use Genton's (2002) mechanism G=w*z'
                                           #   where w =ku' is one or more horizontal instance vectors
                                           #   and   z'=kv  is one or more vertical   instance vectors
                                           #   for explicit Mercer kernels (Carrington, 2018)
        K   = (K + p*(1-c**2)) / (2 * (np.abs(c)+1) )
        return K
    #enddef
    
    return lambda U, V: OISVhc_hidden(U, V, a, b, c, d)
#enddef

def ISVg_kernel(a=1, b=0.5, d=0):
    # Returns a 2 input callable kernel for kernel methods like SVM/SVC, SVR
    #
    # This kernel has the same behaviour, and almost the same shape, as the ISVh kernel.
    #
    # Carrington's Insensitive Sigmoid Variant (ISV) kernel using a Gaussian (2018).
    # A kernel designed for classification that has a central region 
    # that is insensitive (i.e. flat and unweighted) which can ignore the effect data near the 
    # center and thus likely near the class boundary, where class noise/error is greatest.
    #
    # It has all the properties of the Mercer Sigmoid (MSig) kernel, as a generalization of it. 
    #
    # a = central plateau: a=1 flat, a>1 positive slope, a<1 negative slope (not recommended)
    #                      a>exp(1) no plateau/inflection (becomes a single S curve)
    # b = width
    # d = horizontal shift
    #
    # Parameter defaults are suitable for standardization to 2 or 3 sigma
    # Can be further generalized to parameter values in each dimension.
    
    def ISVg_hidden(U, V, a, b, d):   # OISVgc kernel (Carrington, 2018)
        
        ISVg_phi = ISVg_function
        p   = U.shape[1]              # p is number of feature dimensions
        ku  = ISVg_phi(U.T, a, b, d)     # matrix as input and output to phi, with
        kv  = ISVg_phi(V.T, a, b, d)     #   ku, kv, U', V' as vertical instance p-vectors
        K   = np.dot(ku.T, kv)/p       # use Genton's (2002) mechanism G=w*z'
                                      #   where w =ku' is one or more horizontal instance vectors
                                      #   and   z'=kv  is one or more vertical   instance vectors    
                                      #   for explicit Mercer kernels (Carrington, 2018)
        return K
    #enddef
    
    return lambda U, V: ISVg_hidden(U, V, a, b, d)
#enddef

def ISVh_kernel(a=1, b=0.5, d=0):
    # Returns a 2 input callable kernel for kernel methods like SVM/SVC, SVR
    #
    # This kernel has the same behaviour, and almost the same shape, as the ISVh kernel.
    #
    # Carrington's Insensitive Sigmoid Variant (ISV) kernel using a hyperbolic tangent (2018).
    # A kernel designed for classification that has a central region 
    # that is insensitive (i.e. flat and unweighted) which can ignore the effect data near the 
    # center and thus likely near the class boundary, where class noise/error is greatest.
    #
    # It has all the properties of the Mercer Sigmoid (MSig) kernel, as a generalization of it. 
    #
    # a = central plateau: a=1 flat, a>1 positive slope, a<1 negative slope (not recommended)
    #                      a>exp(1) no plateau/inflection (becomes a single S curve)
    # b = width
    # d = horizontal shift
    #
    # Parameter defaults are suitable for standardization to 2 or 3 sigma
    # Can be further generalized to parameter values in each dimension.
    
    def ISVh_hidden(U, V, a, b, d):   # OISVgc kernel (Carrington, 2018)
        
        ISVh_phi = ISVh_function
        p   = U.shape[1]                 # p is number of feature dimensions
        ku  = ISVh_phi(U.T, a, b, d)     # matrix as input and output to phi, with
        kv  = ISVh_phi(V.T, a, b, d)     #   ku, kv, U', V' as vertical instance p-vectors
        K   = np.dot(ku.T, kv)/p         # use Genton's (2002) mechanism G=w*z'
                                         #   where w =ku' is one or more horizontal instance vectors
                                         #   and   z'=kv  is one or more vertical   instance vectors
                                         #   for explicit Mercer kernels (Carrington, 2018)
        return K
    #enddef
    
    return lambda U, V: ISVh_hidden(U, V, a, b, d)
#enddef

def OMIcs_kernel(types,type_weight, Oa=1, Ob=0.5, Oc=1, Od=0, Mb=0.5, Md=0, Ia=1, Ib=0.5, Id=0):
    # Returns a 2 input callable kernel for kernel methods like SVM/SVC, SVR
    #
    def OMIcs_hidden(U, V, b, d):
        
        # requires types in numpy array format
#         if !isinstance(types, np.ndarray):
#             raise Error('types not in numpy ndarray format')
#             #e.g, types = np.array(['O','M','O','I'])
            
        typeO = np.where(types == 'O')[0]
        typeM = np.where(types == 'M')[0]
        typeI = np.where(types == 'I')[0]

        # U is a matrix with horizontal p-vectors representing the instances being evaluated process
        # V is a matrix with horizontal p-vectors representing the instances being evaluated process
        # K is the output kernel matrix representing the similarity between each pair of instances in U and V
            
        rows  = np.shape(U)[0]  # get data ready to...
        UO    = U[rows, typeO]   # apply OISVhc kernel to all rows/instances, but only a subset of features/cols
        UM    = U[rows, typeM]   # apply MSig   kernel to all rows/instances, but only a subset of features/cols
        UI    = U[rows, typeI]   # apply ISVh   kernel to all rows/instances, but only a subset of features/cols

        rows  = np.shape(V)[0]  # get data ready to...
        VO    = V[rows, typeO]   # apply OISVhc kernel to all rows/instances, but only a subset of features/cols
        VM    = V[rows, typeM]   # apply MSig   kernel to all rows/instances, but only a subset of features/cols
        VI    = V[rows, typeI]   # apply ISVh   kernel to all rows/instances, but only a subset of features/cols

        KO    = OISVhc_kernel(Oa, Ob, Oc, Od)   # get kernel for O type
        KM    = MSig_kernel(Mb, Md)             # get kernel for M type
        kI    = ISVh_kernel(Ia, Ib, Id)         # get kernel for I type
                
        KOsubmatrix = type_weight[0] * KO(UO, VO) # apply kernels and weights
        KMsubmatrix = type_weight[1] * KM(UM, VM)
        KIsubmatrix = type_weight[2] * KI(UI, VI)
 
#         def reconstruct_Matrix(subM,idx,fullM_shape):
#             # reconstruct full matrix (all cols) for sub matrix (some cols) 
#             K   = np.zeros(fullM_shape)
#             s   = 0        # s, subM  index
# #             for f in idx:  # f, fullM index
# #                 K = FIXME!
#             return K
        
#         full_shape= [np.shape(U)[1],np.shape(V.T)[0]]
#         K         =     reconstruct_matrix(KO_sub_matrix,typeO,full_shape)
#         K         = K + reconstruct_matrix(KM_sub_matrix,typeM,full_shape)
#         KO_matrix = reconstruct_matrix(KO_sub_matrix,typeO,Mshape)

            
       #K   = KO(U(types[0][:]),)
        MSig_phi = lambda x, b, d: np.tanh((x-d)/b)  # Sigmoid basis function, vectorized

        p   = np.shape(U)[1]       # p is number of feature dimensions
        ku  = MSig_phi(U.T, b, d)  # matrix as input and output to phi, with
        kv  = MSig_phi(V.T, b, d)  #   ku, kv, U', V' as vertical instance p-vectors
        K   = np.dot(ku.T, kv)     # use Genton's (2002) mechanism G=w*z'
                                   #   where w =ku' is one or more horizontal instance vectors
                                   #   and   z'=kv  is one or more vertical   instance vectors
                                   #   for explicit Mercer kernels (Carrington, 2018)
        K   = K / p                # normalized over p dimensions
        return K
    #enddef

    return lambda U, V: OMIcs_hidden(U, V, b, d)
#enddef

def MSig_hidden(U, V, b, d):  # Mercer sigmoid kernel (Carrington et al, 2014)
    # for explanation see MSig_kernel
    MSig_phi = lambda x,b,d: np.tanh((x-d)/b) # Sigmoid basis function, vectorized

    p   = U.shape[1]           # p is number of feature dimensions
    ku  = MSig_phi(U.T, b, d)  # matrix as input and output to phi, with
    kv  = MSig_phi(V.T, b, d)  #   ku, kv, U', V' as vertical instance p-vectors
    K   = np.dot(ku.T, kv)     # use Genton's (2002) mechanism G=w*z'
    #   where w =ku' is one or more horizontal instance vectors
    #   and   z'=kv  is one or more vertical   instance vectors
    #   for explicit Mercer kernels (Carrington, 2018)
    K   = K / p                # normalized over p dimensions
    return K
#enddef

def MSig0_kernel(b=0.5):       # wrapper for Carrington's MSig kernel
    d = 0
    return lambda U, V: MSig_hidden(U, V, b, d)
#enddef

def MSig_kernel(b=0.5, d=0):       # wrapper for Carrington's MSig kernel
    # Returns a 2 input callable kernel for kernel methods like SVM/SVC, SVR
    #
    # b = width
    # d = horizontal shift
    #
    # Parameter defaults are suitable for standardization to 2 or 3 sigma
    # Can be further generalized to parameter values in each dimension.
    return lambda U, V: MSig_hidden(U, V, b, d)
#enddef

def SigN_kernel(a=1, r=-0.1):      # wrapper for normalized sigmoid kernel   
    # Returns a 2 input callable kernel for kernel methods like SVM/SVC, SVR
    #
    # Parameter defaults are suitable for standardization to 2 or 3 sigma
    
    def SigN_hidden(U, V, a, r):      # Normalized sigmoid kernel (Carrington et al, 2014)
        rootp = np.sqrt(U.shape[1])   # p=U.shape[1] is number of feature dimensions
        K     = np.tanh(a*np.dot(U/rootp,V.T/rootp)+r)
        return K
    #enddef

    return lambda U, V: SigN_hidden(U, V, a, r)
#enddef

def Pwr1_kernel():
    return lambda U, V: Pwr_kernel_hidden(U, V, b=1)
#enddef

def Pwr_kernel_hidden(U, V, b=1):  # wrapper for power kernel
    import numpy as np
    # Returns a 2 input callable kernel for kernel methods like SVM/SVC, SVR
    U_instances = U.shape[0]
    V_instances = V.shape[0]
    K           = np.zeros((U_instances, V_instances))
    if   b == 2:
        k= lambda x, y:                   -np.sum((x-y)**2)
    elif b == 1:
        k= lambda x, y:          -np.sqrt(np.sum((x-y)**2))
    else:
        k= lambda x, y: -np.power(np.sqrt(np.sum((x-y)**2)), b)
    #endif
    for Urow in range(U_instances):
        for Vrow in range(V_instances):
            K[Urow, Vrow] = k(U[Urow, :], V[Vrow, :])
        #endfor
    #endfor
    return K
#enddef

def Log1_kernel():
    return lambda U, V: Log_kernel_hidden(U, V, b=1)
#enddef

def Log_kernel_hidden(U, V, b=1):  # wrapper for power kernel
    import numpy as np
    # Returns a 2 input callable kernel for kernel methods like SVM/SVC, SVR
    U_instances = U.shape[0]
    V_instances = V.shape[0]
    K           = np.zeros((U_instances, V_instances))
    if   b == 2:
        k = lambda x, y: -np.log(1 +                  np.sum((x-y)**2))
    elif b == 1:
        k = lambda x, y: -np.log(1 +          np.sqrt(np.sum((x-y)**2)))
    else:
        k = lambda x, y: -np.log(1 + np.power(np.sqrt(np.sum((x-y)**2)), b))
    #endif
    for Urow in range(U_instances):
        for Vrow in range(V_instances):
            K[Urow, Vrow] = k(U[Urow, :], V[Vrow, :])
        #endfor
    #endfor
    return K
#enddef
