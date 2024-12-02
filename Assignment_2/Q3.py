# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:42:25 2024

@author: thoma
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import scipy


#part a
x_i = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
y_i = np.array([-1.22,-3.28, -2.52, 3.74, 3.01, -1.80, 2.49, 5.48, 0.42, 4.80,4.22])

sigmax_i = 1*np.ones(x_i.size) #err on each x
sigmay_i = 2*np.ones(y_i.size) #err on each y

# x_i = np.array([10.,20.,30.,40.,50.,60.,70.,80.])
# y_i = np.array([0.387,5.045,7.299,6.870,16.870,13.951,16.781,20.323])
# sigmax_i = 3.0*np.ones(x_i.size)
# sigmay_i = 4.0*np.ones(y_i.size)
#treat the sigmas as the error on each data point
plt.errorbar(x_i, y_i, xerr = sigmax_i, yerr = sigmay_i, fmt = 'o')
plt.title("plot of y vs x with error bars")
plt.xlabel("x")
plt.ylabel("y")

#part b
#we need to male a 

a = np.linspace(-5.,5.,200) # This is the x-intercept
b = np.linspace( 0., 1.3, 200) # This is the slope ~ 6/10 = 0.6. The data seems to indicate a positive slope 
avals,bvals = np.meshgrid(a,b)

def get_p_D_given_MABI( xi, yi, sigmaxi, sigmayi, A, B):
    N = len(xi)
    norm = (2*np.pi)**(-N/2)  # why not normalized?
    gsum = 0.0
    for i in range(N):
        norm /= np.sqrt(( sigmayi[i]**2 + (B*sigmaxi[i])**2 ) )
        gsum += -(yi[i]-(A+B*xi[i]))**2/(2*(sigmayi[i]**2+(B*sigmaxi[i])**2))
    return norm * np.exp(gsum)

#getting joint PDF since they are directly proportional to the likelihood.
#it needs to be normailized however? to make it the marignal PDF

p_ab_given_DMI = get_p_D_given_MABI(x_i, y_i, sigmax_i, sigmay_i, avals, bvals)
igral = np.trapz(np.trapz(p_ab_given_DMI,a), b)

p_ab_given_DMI /= igral #normalize to get the joint probability function



#Contour plot
plt.figure()
plt.contour( a, b, p_ab_given_DMI)
plt.xlim()
plt.xlabel('a')
plt.ylabel('b')
plt.title('p(a,b|D,M,I)')
plt.colorbar()
plt.show()

#partc: maximize the function by inputing the inverse of the minimum
initial_guess = [1.5,0.6]

#part c. 
def fun( x0, xi, yi, sigmaxi, sigmayi ):
    return (-1)*get_p_D_given_MABI( xi, yi, sigmaxi, sigmayi, x0[0], x0[1] )/igral

result = scipy.optimize.minimize(fun, initial_guess, args=(x_i, y_i, sigmax_i, sigmay_i) )
print(result)
# From the result we get the errorbars
a0best = result.x[0]
a1best = result.x[1]
# The covariance matrix is the inverse of the hessian sigma^2 = H^-1
a0err = np.sqrt( result.hess_inv[0][0] )
a1err = np.sqrt( result.hess_inv[1][1] )
print('a0 =',np.round(a0best,2),'+-',np.round(a0err,2) )
print('a1 =',np.round(a1best,2),'+-',np.round(a1err,2) )

#part d
plt.figure()
x = np.linspace(-10,10,200)
y = np.linspace(-10,10,200)

plt.errorbar(x_i, y_i, xerr = sigmax_i, yerr = sigmay_i, fmt = 'o')
plt.plot(x, a1best*x + a0best)
plt.title("plot of y vs x with error bars")
plt.xlabel("x")
plt.ylabel("y")
plt.text(-10,6, "best fit: y = 0.66x + 1.39")



# part e
#marginalize over b to get a
#this integral is like the table method as prescirbed in the problem
p_A_given_DMI = np.trapz( p_ab_given_DMI/igral, b, axis=0 )
# get normalization
ival = np.trapz( p_A_given_DMI, a )
p_A_given_DMI /= ival

print(np.trapz(p_A_given_DMI,a))
# p_D_given_MABI /= ival
plt.figure(figsize=(3,3),dpi=100)
plt.plot( a, p_A_given_DMI )
plt.title('p(A|D,M,I)')

#marg over a to get b

p_B_given_DMI = np.trapz( p_ab_given_DMI/igral, a, axis=1 ) 

ival = np.trapz( p_B_given_DMI, b )
p_B_given_DMI /= ival
plt.figure(figsize=(3,3),dpi=100)
plt.plot( b, p_B_given_DMI )
plt.title('p(B|D,M,I)')
plt.show()
print(np.trapz(p_B_given_DMI,b)) #check norm


def find_95_percent( x, pdf ):
    
    
    sorted_indices = pdf.argsort()[::-1] # reversed
    xsrt = x.ravel()[sorted_indices]
    pdfsrt = pdf.ravel()[sorted_indices]
    in_95_pct = []
    pct = 0.0
    dx = x[1]-x[0]
    for i in range(len(xsrt)):
        in_95_pct.append( xsrt[i] )
        pct += pdfsrt[i]*dx
        if pct > 0.95:
            break
    minval = np.min(in_95_pct)
    maxval = np.max(in_95_pct)
    best = x[ np.argmax(pdf) ]
    return (minval, best, maxval)
(Amin, Abest, Amax) = find_95_percent( a, p_A_given_DMI)
(Bmin, Bbest, Bmax) = find_95_percent( b, p_B_given_DMI)
print('95% conf interval for A is (', np.round(Amin,2),',',np.round(Amax,2),')')
print('95% conf interval for B is (',np.round(Bmin,2),',',np.round(Bmax,2),')' )
Aerrp = Amax - Abest
Aerrm = Abest - Amin
Berrp = Bmax - Bbest
Berrm = Bbest - Bmin
print('A = ', np.round(Abest,2),'+',np.round(Aerrp,2),'-',np.round(Aerrm,2))
print('B = ', np.round(Bbest,2),'+',np.round(Berrp,2),'-',np.round(Berrm,2))



plt.errorbar( x_i, y_i, xerr=sigmax_i, yerr=sigmay_i, fmt='.',label='data')
plt.plot( [x_i[0],x_i[-1]], [Abest+Bbest*x_i[0],Abest+Bbest*x_i[-1]],label='best fit')
plt.xlabel('x')
plt.ylabel('y')
plt.text(-6,6, "best fit: y = 0.66x + 1.38")
plt.legend()





