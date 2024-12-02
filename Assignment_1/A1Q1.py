# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:53:32 2024

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import math 

data = np.genfromtxt("A1Q1.txt", delimiter = ',', skip_header= 3, skip_footer = 1)
# print(data)


#calculate number of bins using sturges rule and round up to nearest integer
bins = math.ceil(1 + math.log2(len(data)))

plt.hist(data, bins=bins, color='blue', edgecolor='black')
plt.title('Histogram of Data ')
plt.xlabel('Data values')
plt.ylabel('Frequency')

plt.show()


#calculations (b)

mean_data = np.mean(data)
median_data = np.median(data)
mode_data = stat.mode(data)
print(mean_data, median_data, mode_data)
#calculations with binned data
counts, bin_edges = np.histogram(data, bins=bins)     #outputs the counts in each bin, and the edge of each of our 8 bins
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2    #gets the bin center by averaging the bin edges
mean_binned = np.average(bin_centers, weights=counts) #uses each bin center weighted by counts/bin to get the mean 
median_binned = np.median(bin_centers)                # gets the median based on the bin center values. Could maybe do with edges also?
mode_binned = bin_centers[np.argmax(counts)]          #just finds the fullest bin and assigns the median the center value of that bin
print(mean_binned, median_binned, mode_binned)

#c getting stdv and FWHM
std_dev = np.std(data)
FWHM = 2.35 * std_dev  #gaussian expectation of FWHM



#now the binned version
half_max = np.max(counts)/2   #half the max value is approx half the height of fullest bin
#then we need to figure out which bin holds this value
print(half_max)
# we visually see that the second and 7th bins are closest to a height of seven on each side of the mean
#so subtract the bin edges of each of these to get the FWHM
FWHM_binned = bin_edges[6]-bin_edges[1]
print(FWHM,  FWHM_binned)
#They do conform well
print(FWHM_binned/std_dev)
print(bin_edges[5])

