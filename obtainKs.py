import kTools as ks
import autoDiffTools as dif
from datetime import datetime
from numpy import exp, array, zeros

"""
@author: Alberto Pérez-Cervera, 2023

Companion code to the paper

Global phase-amplitude description of oscillatory dynamics via the parameterization method

by A. Pérez-Cervera, Tere M.Seara and Gemma Huguet from Universitat Politècnica de Catalunya
"""

def sigmoid(x, thetaX, aX):
	return 1./(1+exp((x+thetaX)/aX))

def tauH(v):
	a = 0.128*exp(-(v+46.)/18.); b = 4./(1+exp(-(v+23.)/5.))
	return 1./(a+b)

def tauR(v):
	return (28 + exp(-(v+25)/10.5))

def derSigmoid(x, thetaX, aX):
	expFactor = exp((x+thetaX)/aX)
	return (-expFactor)/(((1+expFactor)**2)*aX)

def tauHder(v):
	a = 0.128*exp(-(v+46)/18.); b = 4/(1+exp(-(v+23)/5.))
	aPrima = 0.128*exp(-(v+46)/18.)*(-1/18.)
	bPrima = 4*derSigmoid(v, 23, -5)
	return -(aPrima + bPrima)/((a+b)**2)

def tauRder(v):
	return exp(-(v+25)/10.5)*(-1/10.5)

def myJacobianMatrix(v, h, r, vNa, gT, vK, vL, Iapp, gNa, vT, gL, gK):
    
    hInf = sigmoid(v, 41., 4.); rInf = sigmoid(v, 84., 4.); mInf = sigmoid(v, 37., -7.); pInf = sigmoid(v, 60., -6.2)
    mInfDer = derSigmoid(v, 37., -7.); pInfDer = derSigmoid(v, 60., -6.2); hInfDer = derSigmoid(v, 41., 4.); rInfDer = derSigmoid(v, 84., 4.)
    J = zeros((3,3))
    J[0,0] = -gL -gNa*h*((v-vNa)*3*(mInf**2)*mInfDer + (mInf**3)) - gK*((.75*(1-h))**4) - gT*r*((2*pInf)*(v-vT)*pInfDer + (pInf**2)); 
    J[0,1] = -gNa*(mInf**3)*(v-vNa) - gK*4*((.75*(1-h))**3)*(v - vK)*(-0.75); J[0,2] = -gT*(pInf**2)*(v-vT)
    J[1,0] = (hInfDer*tauH(v) - tauHder(v)*(hInf - h))/tauH(v)**2; J[1,1] = -1/tauH(v); J[1,2] = 0
    J[2,0] = (rInfDer*tauR(v) - tauRder(v)*(rInf - r))/tauR(v)**2; J[2,1] = 0; J[2,2] = -1/tauR(v)
    
    return J

def myVectorField(v, h, r, vNa, gT, vK, vL, Iapp, gNa, vT, gL, gK):

	hInf = sigmoid(v, 41., 4.); rInf = sigmoid(v, 84., 4.); mInf = sigmoid(v, 37., -7.); pInf = sigmoid(v, 60., -6.2)
	f = array([Iapp - gL*(v - vL) - gNa*(mInf**3)*h*(v-vNa) - gK*((.75*(1-h))**4)*(v - vK) - gT*(pInf**2)*r*(v-vT),
		(hInf - h)/tauH(v), (rInf - r)/tauR(v)])
	return f


def main():

    startTime = datetime.now()

    # Please define your parameters
    gT, vNa, vK, vL, Iapp, gNa, vT, gL, gK = [5, 50, -90, -70, 5, 3, 0, 0.05, 5]
    # IMPORTANT: define the parameters in myVectorField and myJacobianMatrix in the same order as next args tuple!!
    args = (vNa, gT, vK, vL, Iapp, gNa, vT, gL, gK)
    
    # Number of points in which the phase [0,1) is discretised. We recommend to choose a multiple of 2 
    N = 512*4;
    # Do you want to save your data?
    write = True
    # Please define initial conditions on the LC and its period
    ic = array([-6.650686246876107433, 0.247447142321608970, 0.001756577816438608])
    period = 8.3955501314387355; 
    
    # Max order (included) of the polinomial expansion
    maxOrder = 10
    # Norms for the K_10 and K_01 vectors
    b1 = 0.5; b2 = 0.5
    
    # Modify next line under your risk!
    dX, dY, dZ = ks.integrateTheSystem(ic, args, period, b1, b2, N, maxOrder+1, write, myJacobianMatrix, myVectorField)

    for order in range(2, maxOrder+1):
        print('Working for order %s' % order)      
        '''
        Automatic differentiation for the Rubin-Terman neuron model
        Note that for each ODE we generate a final result (xExpr for the 1st one, yExpr for the 2nd one and zExpr for the 3rd one)
        We just compose the functions for each ODE
        See library 'autoDiffTools.py' for more details
        You will need to generate the right xExpr, yExpr and zExpr for your model
        '''
        na_vTerm = dif.autoMultplt(dif.autoPow(dif.autoSgmd(dX, 37, -7., order), 3, order), dif.sumConstantTerm(dX, -vNa, order), order)
        gK_hTerm = (.75**4)*dif.autoMultplt(dif.autoPow(dif.sumConstantTerm(-dY, 1, order), 4, order), dif.sumConstantTerm(dX,-vK, order), order)
        gT_vTerm = dif.autoMultplt(dif.autoPow(dif.autoSgmd(dX,60,-6.2,order), 2, order), dif.sumConstantTerm(dX,-vT, order), order)
        remanentTerm = dif.sumConstantTerm(-gL*dif.sumConstantTerm(dX, -vL, order), Iapp, order)
        xExpr = remanentTerm -gNa*dif.autoMultplt(dY, na_vTerm, order) - gK*gK_hTerm - gT*dif.autoMultplt(gT_vTerm, dZ, order)
   
        hInf_vTerm = dif.autoSgmd(dX, 41, 4., order) - dY
        tauH_term = 0.128*dif.autoExp(dif.sumConstantTerm(dX, 46, order), -1/18., order) + 4*dif.autoSgmd(dX, 23, -5., order)
        yExpr = dif.autoMultplt(hInf_vTerm, tauH_term, order)

        rInf_term = dif.autoSgmd(dX, 84, 4., order) -dZ
        tauR_term = dif.sumConstantTerm(dif.autoExp(dif.sumConstantTerm(dX, 25, order), -1/10.5, order), 28, order)
        zExpr = dif.autoDiv(rInf_term, tauR_term, order)

        dX, dY, dZ = ks.obtainHigherOrderKs(xExpr, yExpr, zExpr, dX, dY, dZ, write, order)
        
    print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))

    
    # ------ Author's remark -----
    # If this code has been helpful, please be fair and cite
    # Global phase-amplitude description of oscillatory dynamics via the parameterization method
    # Thanks

if __name__ == '__main__':
    main()
