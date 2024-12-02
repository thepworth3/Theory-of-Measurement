# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:43:55 2024

@author: thoma
"""

import matplotlib.pyplot as plt
import scipy.stats as scp
import math

# we can model the meteor shower as a poisson distribution since we don't have a definite signature of measuring a non-meteor




lambdaa = 15.7/2  #per 30 min rate


poisson = [scp.poisson.pmf(k, lambdaa) for k in range(5)] #calculates P less than 1, 2 ... up to 5
# We want less than 5 prob which is the left tail under the distribution
# print(poisson)

P_less_5 = sum(poisson)

print("The probability that there are less than 5 meteors in 30 minutes is", P_less_5, "or", P_less_5*100, "%.")


#Gaussian Approx
# here we treat the rate as the mean and then compute the z-score

mean = lambdaa
stdev = math.sqrt(lambdaa)

#z for xbar<5

z = (5 - mean)/stdev
print("the z-score is", z)

gaussian_prob = scp.norm.cdf(z) #takes zscore as input to gaussian pdf

print("The probability that there are less than 5 meteors in 30 minutes is", gaussian_prob, "or", gaussian_prob*100, "% when using a gaussian approximation.")