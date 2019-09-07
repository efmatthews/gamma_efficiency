#Eric Matthews
#June 24, 2019
#Utilities for FLUFFY Analysis

#Import statements
#-------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from pylab import rcParams
#-------------------------------------------------------------------------------------



#Functions
#-------------------------------------------------------------------------------------
def plot_spectrum(binned,energy_cal=None,display=True,file_out=None,dpi=500,fmt='eps',axis=None,logscale=False,lines=None,labels=None):
	"""
	function to plot binned energy spectra
	binned = binned energy data
	energy_cal = energy calibration to apply to data, if provided plots are given in terms of energy
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

	if( energy_cal != None ):
		for i in range(0,len(x)):
			x[i] = energy_cal_func(x[i],*energy_cal)

	if( (axis != None) and (energy_cal == None) ):
		x = x[axis[0]:axis[1]]
		y = y[axis[0]:axis[1]]
	elif( (axis != None) and (energy_cal != None) ):
		lower = int( energy_cal_inv(axis[0],*energy_cal) )*2 + 1
		upper = int( energy_cal_inv(axis[1],*energy_cal) )*2 + 1
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
	if( energy_cal == None ):
		plt.xlabel( 'Channel No.' )
	else:
		plt.xlabel( 'Energy (keV)' )

	plt.ylabel( 'No. Counts' )
	if( not(file_out == None) ):
		plt.savefig( file_out, dpi=dpi, format=fmt )
	if( display ):
		plt.show()
	plt.clf()



def plot_spectra(binneds,energy_cal=None,display=True,file_out=None,dpi=500,fmt='eps',axis=None,logscale=False,labels=None):
	"""
	function to plot n binned energy spectra
	binned = binned energy data
	energy_cal = energy calibration to apply to data, if provided plots are given in terms of energy
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

		if( energy_cal != None ):
			for i in range(0,len(x)):
				x[i] = energy_cal_func(x[i],*energy_cal)

		if( (axis != None) and (energy_cal == None) ):
			x = x[axis[0]:axis[1]]
			y = y[axis[0]:axis[1]]
		elif( (axis != None) and (energy_cal != None) ):
			lower = int( energy_cal_inv(axis[0],*energy_cal) )*2 + 1
			upper = int( energy_cal_inv(axis[1],*energy_cal) )*2 + 1
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
	if( energy_cal == None ):
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