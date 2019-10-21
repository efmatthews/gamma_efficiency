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
import decimal
import math
from decimal import Decimal
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

def gauss_poly_fit(data,window_left,window_right,guess_override=[None,None,None,None,None,None,None,None,None]):
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
	
	for i in range(0,len(guess_override)):
		if( guess_override[i] != None ):
			guess[i] = guess_override[i]

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

def gauss_int(B0,B1,B2,B3):
	"""
	Function for definite integral of a tailed Gaussian peak
	B0 is magnitude of peak
	B1 is centroid of peak
	B2 is standard deviation of peak
	B3 is the "beta" parameter for the tail, see GF3 Radware
	return is a number of counts (float)
	"""
	gauss = numpy.sqrt(2.0 * pi) * B0 / numpy.sqrt( 1.0 / B2**2.0 )
	tail = 2.0 * B0 * B3 * numpy.exp( -0.5 * (B1**2.0 / B2**2.0) )

	return gauss + tail

def physical_eff1(E,B0,B1,B2,B3,B4):
	"""
	Function to calculate the energy-dependent efficiency
	E is the gamma energy
	B0-B4 are the fit parameters
	return is efficiency at E
	"""
	return B0 * numpy.exp( -B1 * E**B2 ) * ( 1.0 - numpy.exp( -B3 * E**B4 ) )

def physical_eff1_fit(E_data,eff_data,niter=500,Ns=20,guess=[6e-4,4.6,1.32,1.0,1.0],method='differential_evolution'):
	"""
	Function to fit efficiency calibration given by physical_eff1
	E_data = list of gamma energy data
	eff_data = list of efficiency data
	"""
	numpy.random.seed(0)

	#Make an initial guess
	B0_guess = max(eff_data)

	if( method == 'differential_evolution' ):
		bound_by = [ (B0_guess/10.0, B0_guess*10.0), (0.0,5.0), (0.0,5.0), (0.0,5.0), (0.0,5.0) ]
		diff = lambda B: numpy.sum( numpy.divide( ( physical_eff1(E_data,*B) - numpy.array(eff_data) )**2.0 , numpy.array(eff_data) ) )
		B = scipy.optimize.differential_evolution( diff, bound_by ).x
		B = list(B)

	elif( method == 'basinhopping' ):
		#bound_by = ( [0.0,0.0,0.0,0.0,0.0], [0.5,numpy.inf,numpy.inf,numpy.inf,numpy.inf] )
		diff = lambda B: numpy.sum( numpy.divide( ( physical_eff1(E_data,*B) - numpy.array(eff_data) )**2.0 , numpy.array(eff_data) ) )
		B = scipy.optimize.basinhopping( diff, guess, niter=niter ).x 
		B = list(B)

	elif( method == 'shgo' ):
		bound_by = [ (B0_guess/10.0, B0_guess*10.0), (0.0,5.0), (0.0,5.0), (0.0,5.0), (0.0,5.0) ]
		diff = lambda B: numpy.sum( numpy.divide( ( physical_eff1(E_data,*B) - numpy.array(eff_data) )**2.0 , numpy.array(eff_data) ) )
		B = scipy.optimize.shgo( diff, bound_by ).x 
		B = list(B)

	else:
		raise Exception( method + ' is not an option for minimization.' )

	return B

def physical_eff1_err(E,B_phys,B_phys_cov,h=1e-5):
	"""
	Function for general error propagation formula to obtain uncertainty at a point from the covariance matrix
	E is the energy at which to calculate the uncertainty in the efficiency
	B_phys is a list of 5 phys_eff1 fit parameters
	B_phys_cov is covariance matrix of B_phys parameters
	return is uncertainty in efficiency at E
	"""
	B0,B1,B2,B3,B4 = B_phys
	ders = []
	ders.append( numpy.exp( -B1 * E**B2 ) * ( 1.0 - numpy.exp( -B3 * E**B4 ) ) )
	ders.append( -B0 * E**B2 * numpy.exp( -B1 * E**B2 ) * ( 1.0 - numpy.exp( -B3 * E**B4 ) ) )
	ders.append( -B0 * B1 * E**B2 * numpy.log(E) * numpy.exp( -B1 * E**B2 ) * ( 1.0 - numpy.exp( -B3 * E**B4 ) ) )
	ders.append( B0 * E**B4 * numpy.exp( -(B1 * E**B2) - (B3 * E**B4) ) )
	ders.append( B0 * B3 * E**B4 * numpy.log(E) * numpy.exp( -(B1 * E**B2) - (B3 * E**B4) ) )

	unc = numpy.zeros( (1,len(E)) )[0]
	for i in range(0,5):
		for j in range(0,5):
			unc += ders[i] * ders[j] * B_phys_cov[i,j]
	unc = numpy.sqrt( unc )

	return unc

def physical_eff1_MCerr(E,trials_res_phys):
	"""
	Function to obtain uncertainty via MC at a point from the covariance matrix
	E_ln is the energy at which to calculate the uncertainty in the efficiency
	trials_res_phys is a matrix of MC fitting results
	return is uncertainty in efficiency at E
	"""
	results = numpy.zeros( ( trials_res_phys.shape[1], len(E) ) )
	for i in range(0,trials_res_phys.shape[1]):
		results[i,:] = physical_eff1( E, *trials_res_phys[:,i] )
	unc = numpy.std( results, axis=0 )

	return unc

def logpoly_eff_err(E_ln,B_poly_cov):
	"""
	Function for general error propagation formula to obtain uncertainty at a point from the covariance matrix
	E_ln is the natural log of the energy at which to calculate the uncertainty in the efficiency
	B_poly_cov is covariance matrix of B_poly parameters
	return is uncertainty in efficiency at E
	Warning: this uses the general error propagation formula which is highly insufficient for
	rapidly changing functions much as this.
	"""
	ders = []
	ders.append( numpy.ones( (1,len(E_ln)) )[0] )
	ders.append( E_ln )
	ders.append( numpy.power(E_ln,2.0) )
	ders.append( numpy.power(E_ln,3.0) )
	ders.append( numpy.power(E_ln,4.0) )

	unc = numpy.zeros( (1,len(E_ln)) )[0]
	for i in range(0,5):
		for j in range(0,5):
			unc += ders[i] * ders[j] * B_poly_cov[i,j]
	unc = numpy.sqrt( unc )

	return unc

def logpoly_eff_MCerr(E_ln,trials_res_poly):
	"""
	Function to obtain uncertainty via MC at a point from the covariance matrix
	E_ln is the natural log of the energy at which to calculate the uncertainty in the efficiency
	trials_res_poly is a matrix of MC fitting results
	return is uncertainty in efficiency at E
	"""
	results = numpy.zeros( ( trials_res_poly.shape[1], len(E_ln) ) )
	for i in range(0,trials_res_poly.shape[1]):
		results[i,:] = numpy.polyval( trials_res_poly[:,i], E_ln )
	unc = numpy.std( results, axis=0 )

	return unc

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



def float_to_decimal(f):
    
    # http://docs.python.org/library/decimal.html#decimal-faq
    "Convert a floating point number to a Decimal with no loss of information."

    n, d = f.as_integer_ratio()
    numerator, denominator = Decimal(n), Decimal(d)
    ctx = decimal.Context(prec=60)
    result = ctx.divide(numerator, denominator)

    while ctx.flags[decimal.Inexact]:
        ctx.flags[decimal.Inexact] = False
        ctx.prec *= 2
        result = ctx.divide(numerator, denominator)

    return result

def round_sf(number, sigfig):

    # http://stackoverflow.com/questions/2663612/
    # nicely-representing-a-floating-point-number-in-python/2663623#2663623
    "Round a number to a certain amount of significant digits."

    # First, it must be made certain that the number of significant digits to
    # be rounded to is greater than zero
    assert(sigfig > 0)

    # The number is then converted to a decimal number
    try:
        d=decimal.Decimal(number)
    except TypeError:
        d=float_to_decimal(float(number))
        
    # Separate the number given into three strings: a sign (0 for +/1 for -),
    # the digits as an array, and the exponent.
    sign,digits,exponent = d.as_tuple()
    
    # Case 1: if the amount of digits is less than the amount of significant
    # digits desired, we must extend it by adding zeros at the end of the
    # number until we have the amount of digits desired.
    if len(digits) < sigfig:
        digits = list(digits)
        digits.extend([0] * (sigfig - len(digits)))
    
    # Create a variable "shift" and set it equal to the new exponent of the
    # number after adding zeros.
    shift = d.adjusted()
    
    # Set a new variable, result, equal to an integer of all numbers up to
    # the desired number of digits.
    result=int(''.join(map(str, digits[:sigfig])))
    
    # Round the result if the number of digits is greater than that desired.
    # Look at the number to the right of the last desired digit and decide
    # if rounding up is necessary. Since "result" does not contain a decimal,
    # we can simply add 1 to the current value of result.
    if len(digits) > sigfig and digits[sigfig] >= 5: result += 1
    
    # Set result to a list of strings of its digits (an array)
    result = list(str(result))
    
    # Rounding can change the length of result.
    # If so, adjust shift by the difference between the length of the new
    # result and the amount of sigfigs desired.
    shift += len(result) - sigfig
    
    # Reset length of result to sigfig; chip off the extra digit if we may
    # have just added one on. Otherwise, result remains unchanged..
    result = result[:sigfig]
    
    if shift >= sigfig - 1:
        # Tack more zeros on the end if we shortened the number by rounding it
        # This occurs if the number. 
        result += ['0'] * (shift-sigfig+1)
    elif 0 <= shift:
        # Place the decimal point in between digits if our number contained
        # digits after the decimal
        result.insert(shift+1, '.')
    else:
        # Tack zeros on the front if our number was less than zero originally
        assert(shift < 0)
        result = ['0.'] + ['0']*(-(shift+1)) + result

    if sign:
        result.insert(0, '-')

    return ''.join(result)

def round_unc(number, unc, **kwds):
    
    """Round an uncertainty value to 1 or 2 sigfigs and output the number
    rounded to its uncertainty"""

    if 'form' in kwds.keys():
        outputform = kwds['form']
    else:
        outputform = 'SI'

    if 'crop' in kwds.keys():
        sigfig_crop = kwds['crop']
    else:
        sigfig_crop = 0

    number = str(number)

    if Decimal(number) < 0:
        nsign = 1
    else:
        nsign = 0

    if Decimal(unc) < 0:
        return 'WARNING:: UNCERTAINTY LESS THAN ZERO. TRY AGAIN'

    # Search for scientific notation markers (FORTRAN and Maple compatible),
    # and replace them with 'e'
    number = str(number).replace('E', 'e').replace('D', 'e').replace('Q', 'e')\
                        .replace('-','')
    unc = str(unc).replace('E', 'e').replace('D', 'e').replace('Q', 'e')

    # Create a new number that is the uncertainty to two significant digits,
    # replacing zeros, decimals, and negative signs with blank spaces
    unc_tosigfigz = round_sf(unc, 2).replace('0', '').replace('.', '')\
                                    .replace('-', '')

    if len(unc_tosigfigz) == 1:
        unc_tosigfigz += "0" # We want two digits in our uncertainty

    # If the uncertainty is now greater than 30, read only one significant
    # figure.
    # Now change the uncertainty to include only the correct amount of
    # significant digits.
    if Decimal(unc_tosigfigz) >= Decimal(sigfig_crop): 
        unc_sigfigz=1
        unc_tosigfigz=round_sf(unc_tosigfigz,unc_sigfigz).replace('0','')
    else: # Otherwise, read two
        unc_sigfigz=2
        unc_tosigfigz=round_sf(unc_tosigfigz,unc_sigfigz)

    unc = round_sf(unc, unc_sigfigz)

    # Find the decimal place in both the uncertainty and the number.
    # When all is said and done, we want both numbers to have the same
    # amount of digits in relation to the decimal place ?? **REVISE**
    uncdecplace = unc.find('.')
    numdecplace = number.find('.')

    # First case; if the uncertainty is a float value
    if uncdecplace != -1:
        uncafterdec = len(list(unc)[uncdecplace+1:])

        # Case 1a; the uncertainty is a float, but the number is an
        # integer. Need to add a decimal and zeros for rounding and
        # reset the numdecplace variable.
        if numdecplace==-1:
            number=''.join(list(number)+['.']+['0']*(uncafterdec+1))
            numdecplace = number.find('.')

        # Case 1b; the uncertainty and number are both floats. Need
        # to add only zeros to the number for rounding.
        else:
            number=''.join(list(number)+['0']*uncafterdec)
            numdecplace = number.find('.')

        # Since the uncertainty is a float, it will have digits
        # after the decimal place.
        result = ''.join(number[:numdecplace + uncafterdec + 1])\
                   .replace('.', '')
        result = list(result)

        i = 0
        zeros = 0
        if result[0] == '0':
            for i in range(len(result)-1):
                if result[i] == '0':
                    zeros = zeros + 1
                    i = i + 1
                else:
                    break

        result = ''.join(result)
        # Doing some rounding in the number
        if int(list(number)[numdecplace + uncafterdec + 1]) >= 5:
            result = '0'*zeros + str(int(result) + 1)

        result = list(result)

        numafterdec = len(list(result)[numdecplace + 1:])

        # If we've shortened the number by rounding it, add on a zero
        # to match the amount of digits after the decimal in the
        # uncertainty.
        if int(numafterdec) < int(uncafterdec):
            result += ['0']*(int(uncafterdec) - int(numafterdec) - 1)

        # Replacing negative sign if needed
        result.insert(numdecplace, '.')
        if nsign:
            result.insert(0,'-')

        result = ''.join(result)
        
        return result

    # If the uncertainty has no decimal place
    elif uncdecplace == -1:

        # Adding a decimal to the uncertainty
        unc = ''.join(list(unc) + ['.'])

        # Locate decimal place in the number and uncertainty
        if numdecplace == -1:
            number = ''.join(list(number) + ['.'])
            numdecplace = number.find('.')

        uncdecplace = unc.find('.')

        if unc_sigfigz == 2:
            # Count trailing zeros
            zeros = len(list(unc)[2:uncdecplace])

            # Trimming result to correct place
            result = ''.join(list(number)[:numdecplace - zeros])
            number = ''.join(list(number)[:numdecplace - zeros + 2])\
                       .replace('.','') + '0'*5

            # Round the number if necessary
            if int(number[numdecplace - zeros]) >= 5:
                result = str(int(result) + 1)
            else:
                result = str(result)

            # Insert trailing zeros again
            result = list(result[:numdecplace - zeros])
            result += ['0']*int(zeros)
            if nsign:
                result.insert(0,'-')
            result = ''.join(result)

            return result

        elif unc_sigfigz == 1:
            zeros = len(list(unc)[1:uncdecplace])

            result = ''.join(list(number)[:numdecplace - zeros])\
                       .replace('.', '')

            numnodec = ''.join(list(number)).replace('.', '')
            numnodec += '000'

            if int(list(numnodec)[numdecplace - zeros]) >= 5:
                result = str(int(result) + 1)

            result = list(result)
            result += ['0']*int(zeros)
            if nsign:
                result.insert(0,'-')
            result = ''.join(result)

            return result
#-------------------------------------------------------------------------------------