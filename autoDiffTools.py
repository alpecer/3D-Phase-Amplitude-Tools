from numpy import zeros, exp

'''
Automatic differentiation routines
for more details, see Appendix B in
Global phase-amplitude description of oscillatory dynamics via the parameterization method
'''

def autoSgmd(dX, a, b, maxOrder):

    '''
    automatic differentiation of the sigmoidal function 1/[1+exp( (x+a)/b )]
    '''

    expTerms = zeros(dX.shape); dX = sumConstantTerm(dX, a, maxOrder)/b; expTerms[0,0] = exp(dX[0,0]); 

    expTerms = autoExp(dX, 1, maxOrder+1)
    expTerms = sumConstantTerm(expTerms, 1, maxOrder+1)
    divTerms = zeros(dX.shape); divTerms[0,0] = 1/expTerms[0,0]
    
    for k in range(1, maxOrder+1):
        for j in range(0, k):
            for m in range(0, k-j+1):
                for n in range(0, j+1):
                    divTerms[k-m-n, n+m] = divTerms[j-n, n]*expTerms[k-j-m, m] + divTerms[k-m-n, n+m]
        for l in range(0, k+1):
            divTerms[k-l, l] = (-divTerms[k-l, l])/expTerms[0,0]
    
    return divTerms

def sumConstantTerm(dX, constant, maxOrder):
    
    '''
    This function just sums a constant term to a series
    '''
    sumTerms = zeros(dX.shape)    
    for k in range(0, maxOrder+1):
        for j in range(0, maxOrder+1):
            sumTerms[k,j] = dX[k,j]
    sumTerms[0,0] = dX[0,0] + constant
    return sumTerms

def autoExp(dX, constant, maxOrder):

    '''
    Automatic differentiation of an exponential function exp(constant*x)
    '''
    expTerms = zeros(dX.shape); dX = dX*constant; expTerms[0,0] = exp(dX[0,0]); 
    for k in range(1, maxOrder+1):
        for j in range(0, k):
            for m in range(0, k-j+1):
                for n in range(0, j+1):
                    expTerms[k-m-n, n+m] = (k-j)*expTerms[j-n, n]*dX[k-j-m, m]/k + expTerms[k-m-n, n+m]
                        
    return expTerms

def autoPow(dX, power, maxOrder):
    
    '''
    Automatic differentiation of x**(power) 
    '''
    powTerms = zeros(dX.shape); powTerms[0,0] = dX[0,0]**(power);
    
    for k in range(1, maxOrder+1):
        for j in range(0, k):
            for m in range(0, k-j+1):
                for n in range(0, j+1):
                    powTerms[k-m-n, n+m] = (power*(k-j) - j)*powTerms[j-n, n]*dX[k-j-m, m]/(k*dX[0,0]) + powTerms[k-m-n, n+m]
    return powTerms
    
    
def autoMultplt(a, b, maxOrder):

    '''
    Automatic differentiation of a(x)*b(x)
    '''    
    multTerms = zeros(a.shape);

    for k in range(0, maxOrder+1):
        for j in range(0, k+1):
            for m in range(0, maxOrder+1):
                for n in range(0, maxOrder-m+1):
                    elementOrder = k + n + m
                    if elementOrder >= maxOrder+1:
                        pass
                    else:
                        multTerms[k-j+m, j+n] = a[k-j, j]*b[m, n] + multTerms[k-j+m, j+n]
                        
    return multTerms


def autoDiv(dY, dX, maxOrder):

    '''
    Automatic differentiation of dY(x)/dX(x)
    '''  
    divTerms = zeros(dX.shape); divTerms[0,0] = dY[0,0]/dX[0,0]    
    
    for k in range(1, maxOrder+1):
        for j in range(0, k):
            for m in range(0, k-j+1):
                for n in range(0, j+1):
                    divTerms[k-m-n, n+m] = divTerms[j-n, n]*dX[k-j-m, m] + divTerms[k-m-n, n+m]
        for l in range(0, k+1):
            divTerms[k-l, l] = (dY[k-l, l] - divTerms[k-l, l])/dX[0,0]

    return divTerms