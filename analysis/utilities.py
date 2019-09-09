#Eric Matthews
#June 24, 2019
#Utilities for FLUFFY Analysis

#Import statements
#-------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from pylab import rcParams
from math import exp
from math import sqrt
from math import erf
from math import pi
from math import log as ln
from math import isnan
from scipy.special import erfc
import numpy
numpy.random.seed(0)
from scipy.optimize import curve_fit
import scipy.optimize
numpy.warnings.filterwarnings('ignore')
#-------------------------------------------------------------------------------------



#Functions
#-------------------------------------------------------------------------------------
def energy_cal(x,B0,B1):
	"""
	Function for linear energy calibration
	x = ADC channel number
	B0 = y-intercept of calibration
	B1 = slope of calibration
	return is energy that the ADC channel number corresponds to
	"""
	return B0 + B1*x

def energy_cal_inv(E,B0,B1):
	"""
	Function for inverted linear energy calibration
	E = ADC channel number
	B0 = y-intercept of calibration
	B1 = slope of calibration
	return is ADC channel number corresponding to E
	"""
	return (E - B0) / B1

def energy_cal_fit(centroids,energies):
	"""
	Function to fit data to energy_cal
	centroids = the centroid ADC number of peaks
	energies = the known energy of those peaks
	"""
	guess = [ 0.0, (energies[1]-energies[0])/(centroids[1]-centroids[0]) ]
	B,B_cov = curve_fit(energy_cal,centroids,energies,p0=guess)

	return list(B)

def energy_cal_fit_cov(centroids,energies,centroids_unc,energies_unc,trials=1000):
	"""
	Function to obtain MC covariance matrix for energy_cal_fit
	centroids = the centroid ADC number of peaks
	energies = the known energy of those peaks
	centroids_unc = uncertainties in the centroid ADC number of peaks
	energies_unc = uncertainties in the known energy of those peaks
	"""
	results = numpy.zeros( (2,trials) )
	for i in range(0,trials):
		centroids_vard = numpy.random.normal( centroids, centroids_unc )
		energies_vard = numpy.random.normal( energies, energies_unc )
		B = energy_cal_fit( centroids_vard, energies_vard )
		results[:,i] = B
	B_cov = numpy.cov( results )
	return B_cov

def poly5(x,B0,B1,B2,B3,B4):
	"""
	Function for 5-degree polynomial
	x = ADC channel number
	B0 = 0th order polynomial coefficient
	B1-B4 = further polynomial coefficients
	return is value of polynomial function
	"""
	return B0 + B1*x + B2*numpy.power(x,2) + B3*numpy.power(x,3) + B4*numpy.power(x,4)

def gauss_poly(x,B0,B1,B2,B3,B4,B5,B6,B7,B8):
	"""
	Function for a Gaussian peak on top of a linear background
	x = ADC channel number (int)
	B = floats which are parameters of the fit
		B0 is magnitude of peak
		B1 is centroid of peak
		B2 is standard deviation of peak
		B3 is the "beta" parameter for the tail, see GF3 Radware
		B4-B8 are polynomial coefficients to fit to the background
	return is a number of counts from fit (float)
	"""
	gauss = B0*numpy.exp( -0.5*((x-B1)/B2)**2.0 ) + B0*numpy.exp( (x-B1)/B3 ) * erfc( (x-B1)/(sqrt(2.0)*B2) + B2/(sqrt(2.0)*B3)  )
	bg = B4 + B5*x + B6*numpy.power(x,2) + B7*numpy.power(x,3) + B8*numpy.power(x,4)
	return gauss + bg

def gauss_poly_fit(data,window_left,window_right):
	"""
	Function to fit data to gauss_poly
	data = list of ints which is ADC count data
	window_left = the left bound of the window around the peak (int)
	window_right = the right bound of the window around the peak (int)
	"""
	x_data = range( window_left, window_right+1 )
	y_data = data[ window_left: window_right+1 ]
	slope_guess = ( numpy.average( y_data[-3:] ) - numpy.average( y_data[0:3] ) ) / ( x_data[-2] - x_data[1] )
	if( slope_guess < 0.0 ):
		slope_guess = 0.0
	yint_guess = numpy.average( y_data[0:3] ) - slope_guess * x_data[1]
	if( yint_guess < 0.0 ):
		yint_guess = 0.0
	mag_guess = max(y_data)
	centroid_guess = numpy.average( x_data, weights=y_data )
	sigma_guess = sqrt( numpy.average( (x_data-centroid_guess)**2.0, weights=y_data ) )
	beta_guess = sigma_guess/2.0
	guess = [ mag_guess, centroid_guess, sigma_guess, beta_guess, yint_guess, slope_guess, 0.0, 0.0, 0.0 ]
	bound_by = ( [0.0,0.0,0.0,0.0,0.0,-numpy.inf,-numpy.inf,-numpy.inf,-numpy.inf], [numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf] )
	sigmas = numpy.sqrt( y_data )
	sigmas[numpy.isnan(sigmas)]=1.0

	try:
		B,B_cov = list( curve_fit(gauss_poly,x_data,y_data,p0=guess,bounds=bound_by,sigma=sigmas) )
	except RuntimeError as e:
		try:
			B,B_cov = list( curve_fit(gauss_poly,x_data,y_data,p0=guess,bounds=bound_by,sigma=sigmas,maxfev=10000) )
		except RuntimeError as e:
			B = [ 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
	res = list(B)

	return res

def gauss_gauss_poly(x,B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12):
	"""
	Function for a Gaussian peak on top of a linear background
	x = ADC channel number (int)
	B = floats which are parameters of the fit
		B0 is magnitude of peak
		B1 is centroid of peak
		B2 is standard deviation of peak
		B3 is the "beta" parameter for the tail, see GF3 Radware
		B4-B7 are same parameters as B0-B3 but for a second gaussian
		B8-B12 are polynomial coefficients to fit to the background
	return is a number of counts from fit (float)
	"""
	gauss1 = B0*numpy.exp( -0.5*((x-B1)/B2)**2.0 ) + B0*numpy.exp( (x-B1)/B3 ) * erfc( (x-B1)/(sqrt(2.0)*B2) + B2/(sqrt(2.0)*B3)  )
	gauss2 = B4*numpy.exp( -0.5*((x-B5)/B6)**2.0 ) + B4*numpy.exp( (x-B5)/B7 ) * erfc( (x-B5)/(sqrt(2.0)*B6) + B6/(sqrt(2.0)*B7)  )
	bg = B8 + B9*x + B10*numpy.power(x,2) + B11*numpy.power(x,3) + B12*numpy.power(x,4)
	return gauss1 + gauss2 + bg

def gauss_gauss_poly_fit(data,window_left,window_right,guess_override=[None,None,None,None,None,None,None,None,None,None,None,None,None]):
	"""
	Function to fit data to gauss_gauss_poly
	data = list of ints which is ADC count data
	window_left = the left bound of the window around the peak (int)
	window_right = the right bound of the window around the peak (int)
	"""
	x_data = range( window_left, window_right+1 )
	y_data = data[ window_left: window_right+1 ]
	slope_guess = ( numpy.average( y_data[-3:] ) - numpy.average( y_data[0:3] ) ) / ( x_data[-2] - x_data[1] )
	if( slope_guess < 0.0 ):
		slope_guess = 0.0
	yint_guess = numpy.average( y_data[0:3] ) - slope_guess * x_data[1]
	if( yint_guess < 0.0 ):
		yint_guess = 0.0
	mag_guess = max(y_data)
	centroid_guess = numpy.average( x_data, weights=y_data )
	sigma_guess = sqrt( numpy.average( (x_data-centroid_guess)**2.0, weights=y_data ) )
	beta_guess = sigma_guess/2.0
	guess = [ mag_guess, centroid_guess, sigma_guess, beta_guess, mag_guess, centroid_guess+2.0, sigma_guess, beta_guess, yint_guess, slope_guess, 0.0, 0.0, 0.0 ]

	for i in range(0,len(guess_override)):
		if( guess_override[i] != None ):
			guess[i] = guess_override[i]

	bound_by = ( [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-numpy.inf,-numpy.inf,-numpy.inf,-numpy.inf], [numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf] )
	sigmas = numpy.sqrt( y_data )
	sigmas[numpy.isnan(sigmas)]=1.0

	try:
		B,B_cov = list( curve_fit(gauss_gauss_poly,x_data,y_data,p0=guess,bounds=bound_by,sigma=sigmas) )
	except RuntimeError as e:
		try:
			B,B_cov = list( curve_fit(gauss_gauss_poly,x_data,y_data,p0=guess,bounds=bound_by,sigma=sigmas,maxfev=10000) )
		except RuntimeError as e:
			B = [ 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
	res = list(B)

	return res

def gauss_gauss_gauss_poly(x,B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16):
	"""
	Function for a Gaussian peak on top of a linear background
	x = ADC channel number (int)
	B = floats which are parameters of the fit
		B0 is magnitude of peak
		B1 is centroid of peak
		B2 is standard deviation of peak
		B3 is the "beta" parameter for the tail, see GF3 Radware
		B4-B7 are same parameters as B0-B3 but for a second gaussian
		B8-B11 are same parameters as B0-B3 but for a second gaussian
		B12-B16 are polynomial coefficients to fit to the background
	return is a number of counts from fit (float)
	"""
	gauss1 = B0*numpy.exp( -0.5*((x-B1)/B2)**2.0 ) + B0*numpy.exp( (x-B1)/B3 ) * erfc( (x-B1)/(sqrt(2.0)*B2) + B2/(sqrt(2.0)*B3)  )
	gauss2 = B4*numpy.exp( -0.5*((x-B5)/B6)**2.0 ) + B4*numpy.exp( (x-B5)/B7 ) * erfc( (x-B5)/(sqrt(2.0)*B6) + B6/(sqrt(2.0)*B7)  )
	gauss3 = B8*numpy.exp( -0.5*((x-B9)/B10)**2.0 ) + B8*numpy.exp( (x-B9)/B11 ) * erfc( (x-B9)/(sqrt(2.0)*B10) + B10/(sqrt(2.0)*B11)  )
	bg = B12 + B13*x + B14*numpy.power(x,2) + B15*numpy.power(x,3) + B16*numpy.power(x,4)
	return gauss1 + gauss2 + gauss3 + bg

def gauss_gauss_gauss_poly_fit(data,window_left,window_right,guess_override=[None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]):
	"""
	Function to fit data to gauss_gauss_gauss_poly
	data = list of ints which is ADC count data
	window_left = the left bound of the window around the peak (int)
	window_right = the right bound of the window around the peak (int)
	"""
	x_data = range( window_left, window_right+1 )
	y_data = data[ window_left: window_right+1 ]
	slope_guess = ( numpy.average( y_data[-3:] ) - numpy.average( y_data[0:3] ) ) / ( x_data[-2] - x_data[1] )
	if( slope_guess < 0.0 ):
		slope_guess = 0.0
	yint_guess = numpy.average( y_data[0:3] ) - slope_guess * x_data[1]
	if( yint_guess < 0.0 ):
		yint_guess = 0.0
	mag_guess = max(y_data)
	centroid_guess = numpy.average( x_data, weights=y_data )
	sigma_guess = sqrt( numpy.average( (x_data-centroid_guess)**2.0, weights=y_data ) )
	beta_guess = sigma_guess/2.0
	guess = [ mag_guess, centroid_guess, sigma_guess, beta_guess, mag_guess, centroid_guess+2.0, sigma_guess, beta_guess, mag_guess, centroid_guess-2.0, sigma_guess, beta_guess, yint_guess, slope_guess, 0.0, 0.0, 0.0 ]
	
	for i in range(0,len(guess_override)):
		if( guess_override[i] != None ):
			guess[i] = guess_override[i]

	bound_by = ( [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-numpy.inf,-numpy.inf,-numpy.inf,-numpy.inf], [numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf,numpy.inf] )
	sigmas = numpy.sqrt( y_data )
	sigmas[numpy.isnan(sigmas)]=1.0

	try:
		B,B_cov = list( curve_fit(gauss_gauss_gauss_poly,x_data,y_data,p0=guess,bounds=bound_by,sigma=sigmas) )
	except RuntimeError as e:
		try:
			B,B_cov = list( curve_fit(gauss_gauss_gauss_poly,x_data,y_data,p0=guess,bounds=bound_by,sigma=sigmas,maxfev=10000) )
		except RuntimeError as e:
			B = [ 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
	res = list(B)

	return res

def plot_spectrum(binned,energy_cal_in=None,display=True,file_out=None,dpi=500,fmt='eps',axis=None,logscale=False,lines=None,labels=None):
	"""
	function to plot binned energy spectra
	binned = binned energy data
	energy_cal_in = energy calibration to apply to data, if provided plots are given in terms of energy
	display = boolean to decide whether or not to class plt.show()
	file_out = location of output file to write image to
	dpi = resolution of output image
	"""
	x = [ 0 ]
	for i in range(0,len(binned)):
		x.append( i )
		x.append( i+1 )

	y = [ 0 ]
	for i in range(0,len(binned)):
		y.append( binned[i] )
		y.append( binned[i] )

	x.append( len(binned) )
	y.append( 0 )

	if( energy_cal_in != None ):
		for i in range(0,len(x)):
			x[i] = energy_cal(x[i],*energy_cal_in)

	if( (axis != None) and (energy_cal_in == None) ):
		x = x[axis[0]:axis[1]]
		y = y[axis[0]:axis[1]]
	elif( (axis != None) and (energy_cal_in != None) ):
		lower = int( energy_cal_inv(axis[0],*energy_cal_in) )*2 + 1
		upper = int( energy_cal_inv(axis[1],*energy_cal_in) )*2 + 1
		if(lower < 0):
			lower = 0
		x = x[lower:upper]
		y = y[lower:upper]

	rcParams['figure.figsize'] = 11, 8
	font = {'size'   : 16}
	plt.rc('font', **font)
	if( logscale ):
		plt.semilogy( x, y, 'k', linewidth=0.5 )
	else:
		plt.plot( x, y, 'k', linewidth=0.5 )

	if( lines != None ):
		plt.vlines(lines,0,max(binned),color='r',linewidth=0.5)

	if( labels != None ):
		plt.legend()
	if( energy_cal_in == None ):
		plt.xlabel( 'Channel No.' )
	else:
		plt.xlabel( 'Energy (keV)' )

	plt.ylabel( 'No. Counts' )
	if( not(file_out == None) ):
		plt.savefig( file_out, dpi=dpi, format=fmt )
	if( display ):
		plt.show()
	plt.clf()



def plot_spectra(binneds,energy_cal_in=None,display=True,file_out=None,dpi=500,fmt='eps',axis=None,logscale=False,labels=None):
	"""
	function to plot n binned energy spectra
	binned = binned energy data
	energy_cal_in = energy calibration to apply to data, if provided plots are given in terms of energy
	display = boolean to decide whether or not to class plt.show()
	file_out = location of output file to write image to
	dpi = resolution of output image
	"""
	rcParams['figure.figsize'] = 11, 8
	font = {'size'   : 16}
	plt.rc('font', **font)
	for q in range(0,len(binneds)):
		binned = binneds[q]
		x = [ 0 ]
		for i in range(0,len(binned)):
			x.append( i )
			x.append( i+1 )

		y = [ 0 ]
		for i in range(0,len(binned)):
			y.append( binned[i] )
			y.append( binned[i] )

		x.append( len(binned) )
		y.append( 0 )

		if( energy_cal_in != None ):
			for i in range(0,len(x)):
				x[i] = energy_cal_func(x[i],*energy_cal_in)

		if( (axis != None) and (energy_cal_in == None) ):
			x = x[axis[0]:axis[1]]
			y = y[axis[0]:axis[1]]
		elif( (axis != None) and (energy_cal_in != None) ):
			lower = int( energy_cal_inv(axis[0],*energy_cal_in) )*2 + 1
			upper = int( energy_cal_inv(axis[1],*energy_cal_in) )*2 + 1
			if(lower < 0):
				lower = 0
			x = x[lower:upper]
			y = y[lower:upper]

		if( logscale ):
			if( labels != None ):
				plt.semilogy( x, y, label=labels[q], linewidth=0.5 )
			else:
				plt.semilogy( x, y, linewidth=0.5 )
		else:
			if( labels != None ):
				plt.plot( x, y, label=labels[q], linewidth=0.5 )
			else:
				plt.plot( x, y, linewidth=0.5 )

	if( labels != None ):
		plt.legend()
	if( energy_cal_in == None ):
		plt.xlabel( 'Channel No.' )
	else:
		plt.xlabel( 'Energy (keV)' )

	plt.ylabel( 'No. Counts' )
	if( not(file_out == None) ):
		plt.savefig( file_out, dpi=dpi, format=fmt )
	if( display ):
		plt.show()
	plt.clf()
#-------------------------------------------------------------------------------------