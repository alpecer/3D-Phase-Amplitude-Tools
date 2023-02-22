from numpy import zeros, reshape, column_stack, exp, log, dot, where, pi, hstack, arange, fft, mean, around, linspace, array, savetxt
from numpy import linalg as LA
from scipy.integrate import odeint

"""
@author: Alberto Pérez-Cervera, 2023

Companion code to the paper

Global phase-amplitude description of oscillatory dynamics via the parameterization method

by A. Pérez-Cervera, Tere M.Seara and Gemma Huguet from Universitat Politècnica de Catalunya
"""

def get_Q(sol, C, floqExp, timeArray, period):
  
    variationalMatrix = zeros((nDim, nDim))
    Q = []; timeArray = timeArray/period; N = len(timeArray)
    eigenMatrix = zeros((3,3))
    for i in range(0, N):
        variationalMatrix = sol[i].reshape((nDim, nDim))
        eigenMatrix[0,0] = exp(-log(floqExp[0])*timeArray[i]); 
        eigenMatrix[1,1] = exp(-log(floqExp[1])*timeArray[i]);
        eigenMatrix[2,2] = exp(-log(floqExp[2])*timeArray[i])
        qElement = dot(dot(dot(variationalMatrix, C), eigenMatrix), LA.inv(C))
        Q.append(qElement)
    return array(Q)
        
def get_k1(sol, k1_zero, timeArray, landa, period):
  
    variationalMatrix = zeros((nDim, nDim)); N = len(timeArray)
    k1 = zeros((nDim, N))
    for i in range(0, N):
        variationalMatrix = sol[i].reshape((nDim, nDim))
        k1[0, i], k1[1, i], k1[2, i] = dot(variationalMatrix, k1_zero)*exp(-landa*timeArray[i])
    return k1[0,:], k1[1,:], k1[2,:]

def getLandas(floqExp, C, period):

    index = where(abs(floqExp - 1) > 0.01)[0]
    landa1 = log(floqExp[index[0]])/period; v1 = C[:,index[0]]; landa2 = log(floqExp[index[1]])/period; v2 = C[:,index[1]]    
    return v1, v2, landa1, landa2, C

def saveData(data, fileName, delimiter="\t"):

    with open(fileName, 'a') as f:
        savetxt(f, column_stack(data[:]), fmt='%1.3f', delimiter="\t")
    f.close()

def createJacMatrixArray(x, y, z, myJacobianMatrix, N, args):

    jacValues = []; 
    for i in range(0, N):
        jacValues.append(myJacobianMatrix(x[i], y[i], z[i], *args))
    return array(jacValues)


def integrateTheSystem(ic, args, period, b1, b2, N, maxOrder, write, myJacobianMatrix, myVectorField):
    
    global jacobianMatrix, vectorField, constantProduct, invConstantProduct, landa1, landa2, floqExp, totalPeriod, jacMatrix 
    
    t = linspace(0, period, num=N+1, endpoint=True); nDim = len(ic)
    ic = array([ic[0], ic[1], ic[2], 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    
    jacobianMatrix = myJacobianMatrix; vectorField = myVectorField; totalPeriod = period
    
    sol = odeint(myVarSystem, ic, t, args, rtol=2.5e-14, atol=2.5e-14, mxstep = 2000000000, hmax=0.001)
    x = sol[:-1,0]; y = sol[:-1,1]; z = sol[:-1,2]; timeArray = sol[:-1,3]
    checkError_K0(x, y, z, args, period, N, myVectorField)
    dX = zeros((maxOrder+1, maxOrder+1, N)); dY = zeros((maxOrder+1, maxOrder+1, N)); dZ = zeros((maxOrder+1, maxOrder+1, N));
    dX[0, 0, :] = x[:]; dY[0, 0, :] = y[:]; dZ[0, 0, :] = z[:]
    
    monodromyMatrix = sol[-1][(nDim+1):].reshape((nDim, nDim))
     
    floqExp, C = LA.eig(monodromyMatrix); nDim = len(floqExp)
    idx = floqExp.argsort(); floqExp = floqExp[idx]; C = C[:,idx]
    jacMatrix = createJacMatrixArray(x, y, z, myJacobianMatrix, N, args)

    Q = get_Q(sol[:,(nDim+1):], C, floqExp, timeArray, period); 
    v1, v2, landa1, landa2, C = getLandas(floqExp, C, period);
    invConstantProduct = []; constantProduct = []

    for i in range(0, N):
        invConstantProduct.append(dot(LA.inv(C), LA.inv(Q[i])));
        constantProduct.append(dot(Q[i], C));
    constantProduct = array(constantProduct); invConstantProduct = array(invConstantProduct)

    k10_x, k10_y, k10_z = get_k1(sol[:,(nDim+1):], b1*v1, timeArray, landa1, period)
    dX[1, 0, :] = k10_x[:]; dY[1, 0, :] = k10_y[:]; dZ[1, 0, :] = k10_z[:]
    ckeckError_k1(k10_x, k10_y, k10_z, jacMatrix, landa1, period, N)
  
    k01_x, k01_y, k01_z = get_k1(sol[:,(nDim+1):], b2*v2, timeArray, landa2, period)
    dX[0, 1, :] = k01_x[:]; dY[0, 1, :] = k01_y[:]; dZ[0, 1, :] = k01_z[:]
    ckeckError_k1(k01_x, k01_y, k01_z, jacMatrix, landa2, period, N)

    if write: 
        fp = open('kx.dat', 'w'); fp.close(); fp = open('ky.dat', 'w'); fp.close(); fp = open('kz.dat', 'w'); fp.close()   
        saveData(x, 'kx.dat'); saveData(y, 'ky.dat'); saveData(z, 'kz.dat');  
        saveData(k10_x, 'kx.dat'); saveData(k10_y, 'ky.dat'); saveData(k10_z, 'kz.dat');
        saveData(k01_x, 'kx.dat'); saveData(k01_y, 'ky.dat'); saveData(k01_z, 'kz.dat');

    return dX, dY, dZ

def myVarSystem(z, t, *args):
    
    v, h, r, u, aa, bb, cc, dd, ee, ff, gg, hh, ii = z
    f = hstack((vectorField(v, h, r, *args),1))

    J = jacobianMatrix(v, h, r, *args)
    varMatrix = array([aa, bb, cc, dd, ee, ff, gg, hh, ii]).reshape((3,3))

    varProduct = dot(J,varMatrix)

    f = hstack((f,varProduct.reshape(3*3)))
    return f

def compute_kDerivates(ks, N):
    
    derivateArray = 2j*pi*arange(0,N/2 +1)
    coeffs = fft.rfft(ks)
    return fft.irfft(derivateArray*coeffs)
    

def checkError_K0(x, y, z, args, period, N, myVectorField):
    
    xValues = zeros(N); yValues = zeros(N); zValues = zeros(N)

    for i in range(0, N):
        fValueX, fValueY, fValueZ = myVectorField(x[i], y[i], z[i], *args)
        xValues[i] = fValueX; yValues[i] = fValueY; zValues[i] = fValueZ
    
    dx = compute_kDerivates(x, N); dy = compute_kDerivates(y, N); dz = compute_kDerivates(z, N);
    xError = max(abs(dx/period - xValues)); yError = max(abs(dy/period - yValues)); zError = max(abs(dz/period - zValues));
    print('Error k0')
    print('\t xError = %s \t yError = %s \t zError = %s' % (around(xError, 14), around(yError,14), around(zError,14)))
    
def ckeckError_k1(k1_x, k1_y, k1_z, jacMatrix, landa, period, N):
    
    dx = compute_kDerivates(k1_x, N)/period; dy = compute_kDerivates(k1_y, N)/period; dz = compute_kDerivates(k1_z, N)/period;
    xVals = zeros(N); yVals = zeros(N); zVals = zeros(N)
    
    for i in range(0, N): 
        jacProduct = dot(jacMatrix[i], array([[k1_x[i]],[k1_y[i]],[k1_z[i]]]))
        xVals[i] = jacProduct[0][0]; yVals[i] = jacProduct[1][0]; zVals[i] = jacProduct[2][0];
    
    xError = max(abs(dx + landa*k1_x - xVals)); yError = max(abs(dy + landa*k1_y - yVals)); zError = max(abs(dz + landa*k1_z - zVals));

    print('Error k1')
    print('\t xError = %s \t yError = %s \t zError = %s' % (around(xError,14), around(yError,14), around(zError,14)))

def checkError_ks(kx, ky, kz, jacMatrix, factor, bx, by, bz, period, N):  
    
    dx = compute_kDerivates(kx, N)/period; dy = compute_kDerivates(ky, N)/period; dz = compute_kDerivates(kz, N)/period;
    xVals = zeros(N); yVals = zeros(N); zVals = zeros(N)
    
    for i in range(0, N): 
        jacProduct = dot(jacMatrix[i], array([[kx[i]],[ky[i]],[kz[i]]]))
        xVals[i] = jacProduct[0][0]; yVals[i] = jacProduct[1][0]; zVals[i] = jacProduct[2][0];
    
    xError = max(abs(dx + factor*kx - xVals - bx)); 
    yError = max(abs(dy + factor*ky - yVals - by)); 
    zError = max(abs(dz + factor*kz - zVals - bz));

    print('\t xError = %s \t yError = %s \t zError = %s' % (around(xError,14), around(yError,14), around(zError,14)))


def obtainHigherOrderKs(xExpr, yExpr, zExpr, dX, dY, dZ, write, order):
    
    N = xExpr.shape[-1]

    for i in range(0, order+1):
        bX = []; bY = []; bZ = []
        for j in range(0, N):
            winner = dot(invConstantProduct[j], array([[xExpr[order-i, i][j]],[yExpr[order-i, i][j]],[zExpr[order-i, i][j]]]))
            bX.append(winner[0][0]); bY.append(winner[1][0]); bZ.append(winner[2][0]) 

        bX_coeffs = fft.rfft(array(bX)); bY_coeffs = fft.rfft(array(bY)); bZ_coeffs = fft.rfft(array(bZ));
        factor = (order-i)*landa1 + (i)*landa2
        constantFactor = 2j*pi*arange(0,N/2 +1)/totalPeriod + factor
        ux = bX_coeffs/(constantFactor - log(floqExp[0])/totalPeriod); kx = fft.irfft(ux)
        uy = bY_coeffs/(constantFactor - log(floqExp[1])/totalPeriod); ky = fft.irfft(uy)
        uz = bZ_coeffs/(constantFactor - log(floqExp[2])/totalPeriod); kz = fft.irfft(uz)

        kX = []; kY = []; kZ = []
        for j in range(0, N):
            winner = dot(constantProduct[j], array([[kx[j]],[ky[j]],[kz[j]]]))
            kX.append(winner[0][0]); kY.append(winner[1][0]); kZ.append(winner[2][0]) 
            
        dX[order-i, i] = array(kX); dY[order-i, i] = array(kY); dZ[order-i, i] = array(kZ)
        print('Error K%s%s' % (order-i, i))
        checkError_ks(array(kX), array(kY), array(kZ), jacMatrix, factor, array(xExpr[order-i, i]), array(yExpr[order-i, i]), array(zExpr[order-i, i]), totalPeriod, N)
        if write:
            saveData(array(kX), 'kx.dat'); saveData(array(kY), 'ky.dat'); saveData(array(kZ), 'kz.dat');
    
    return dX, dY, dZ

global nDim, jacobianMatrix, vectorField
nDim = 3
