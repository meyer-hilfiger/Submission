# This file verifies that the constraint of Proposition 5.7 of the paper is verified by RLPN + Dumer86 parameters which are available at https://github.com/tillich/RLPNdecoding/blob/master/supplementaryMaterial/RLPN_Dumer86.csv. These parameters correspond to optimal asymptotic parameters for Proposition 3.10 along with Section 4.1 of the paper https://eprint.iacr.org/2022/1000.pdf
from scipy.optimize import brentq
from math import sqrt
from math import log2
import numpy as np
import csv
import subprocess


def H2_G(x,a):
	if(x == 0 or x == 1):
		return -a
	return -x*log2(x) - (1-x)*log2(1-x) - a

# Binary entropy
def H2(x):
	assert x >= 0
	assert x <= 1
	return H2_G(x,0)

# Inverse on [0,1/2] of binary entropy
def H2_I(x):
	if(x < 0 or x > 1):
		assert(False)
	if(x == 0):
		return 0
	if(x == 1):
		return 0.5
	return brentq(H2_G,a=0,b=0.5,args=(x))

# Returns asymptotic complexity exponent of ISD Dumer91 to decode a code of rate 0 < R < 1 
# at relative distance 0 <= tau <= h_2^{-1}(1-R)
# Dumer91 : https://www.researchgate.net/publication/296573348_On_minimum_distance_decoding_of_linear_codes
def isd_dumer91(R,tau,lamb,omega):

	#Condition on the parameters
	assert(omega < min(tau, R + lamb))
	assert(omega > max(0.0,R+lamb+tau-1.0))

	assert(lamb < 1-R)
	assert(lamb > 0)


	# Probability that there exists omega errors (among tau) in an extended information set of size R + lamb
	success_proba = (1.0 -R - lamb)*H2((tau-omega)/(1.0 -R - lamb)) + (R+lamb)*H2(omega/(R+lamb)) - H2(tau)
	# Computing the lists
	list_handling = max(((R+lamb)/2)*H2(omega/(R+lamb)), (R+lamb)*H2(omega/(R+lamb)) - lamb)

	return list_handling - success_proba


# RLPN original parameters when using Dumer86 method to compute low weight parity-checks
# Clone of : https://github.com/tillich/RLPNdecoding/blob/master/supplementaryMaterial/RLPN_Dumer86.csv
# Date : 1 June 2023
parameters_file = "RLPN_Dumer86.csv"

# For each rate R and parameters \sigma, \mu, \omega in "RLPN_Dumer86.csv" the following list contains the parameters used in ISD Dumer91 decoder to decode a code of rate (R-s)/(1-s) at relative distance u/(1-s)
# Parameters were obtained using Cawof optimizer (Library by R. C. Torres available at https://gforge.inria.fr/projects/cawof/)
PARAMS_ISD_DUMER_91 = [[1.12e-06, 5.282e-07], [1.6092e-06, 7.585e-07], [1.2733e-06, 4.829e-07], [1.8221e-06, 6.896e-07], [2.5999e-06, 9.81e-07], [6.9542e-06, 3.2837e-06], [5.3629e-06, 2.0184e-06], [1.43828e-05, 6.7739e-06], [1.11111e-05, 4.1743e-06], [1.5993e-05, 6.0011e-06], [2.30225e-05, 8.6266e-06], [3.31506e-05, 1.24022e-05], [9.24253e-05, 4.38272e-05], [0.0001353369, 6.44477e-05], [9.88195e-05, 3.67186e-05], [0.0001421063, 5.26368e-05], [0.0002041948, 7.53514e-05], [0.0002931271, 0.000107689], [0.0004202502, 0.0001535783], [0.0006014996, 0.0002184397], [0.0008591562, 0.0003096974], [0.0017422001, 0.0006159366], [0.0025730772, 0.000896262], [0.003363787, 0.00115698], [0.0041251142, 0.0014030755], [0.0048594205, 0.0016361458], [0.0055685879, 0.0018574125], [0.0062541599, 0.0020678503], [0.006917414, 0.0022682531], [0.007559438, 0.0024592881], [0.0081811439, 0.0026415124], [0.0087833484, 0.0028154187], [0.0093667431, 0.0029814269], [0.0099319607, 0.0031399169], [0.0104795522, 0.0032912207], [0.0110099613, 0.0034356163], [0.0115237026, 0.0035734027], [0.012021116, 0.0037047983], [0.0125025329, 0.0038300131], [0.0129683565, 0.0039492813], [0.0134188571, 0.0040627773], [0.0138542184, 0.0041706376], [0.0142747406, 0.0042730425], [0.0146807047, 0.0043701602], [0.0150721928, 0.0044620784], [0.0154494397, 0.0045489427], [0.015812651, 0.0046308842], [0.0161619757, 0.0047080105], [0.0164974958, 0.0047804019], [0.0168192873, 0.0048481364], [0.0171276388, 0.0049113681], [0.0174224176, 0.004970095], [0.0177039425, 0.0050244794], [0.0179721161, 0.0050745308], [0.0182271978, 0.0051203869], [0.0184691072, 0.0051620622], [0.0186978337, 0.0051995967], [0.0189135787, 0.0052331042], [0.019116415, 0.0052626517], [0.0193060628, 0.0052881846], [0.0194829717, 0.0053098995], [0.0196468728, 0.0053277453], [0.019797861, 0.0053417959], [0.0199358192, 0.0053520531], [0.0200608826, 0.0053586036], [0.020173052, 0.0053614884], [0.0202723106, 0.0053607428], [0.020358327, 0.0053562999], [0.0204314243, 0.0053483069], [0.0204915296, 0.0053367812], [0.020538213, 0.0053216269], [0.0205720143, 0.005303059], [0.0205924579, 0.0052809683], [0.0205995242, 0.0052553923], [0.020593142, 0.0052263527], [0.020573267, 0.0051938798], [0.0205398094, 0.0051579909], [0.0204927771, 0.0051187334], [0.020431547, 0.0050759644], [0.0203566659, 0.0050298974], [0.020267697, 0.0049804481], [0.0201641123, 0.0049275093], [0.0200464336, 0.004871287], [0.0199138608, 0.004811597], [0.0197665575, 0.004748542], [0.0196044705, 0.0046821607], [0.0194268299, 0.0046122885], [0.0192337634, 0.0045390219], [0.0190247116, 0.0044622639], [0.0188004162, 0.0043822864], [0.018558867, 0.0042985925], [0.0183009571, 0.0042115027], [0.0180259446, 0.0041208843], [0.0177332491, 0.0040266562], [0.0174228326, 0.0039288898], [0.0170940952, 0.0038275085], [0.0172461611, 0.0038547236], [0.0163787307, 0.0036134959], [0.0159914252, 0.0035008885], [0.0155826181, 0.0033842393], [0.0151526363, 0.003263756], [0.0147000788, 0.0031392123], [0.0142232173, 0.0030103199], [0.0137226268, 0.002877384], [0.0131953491, 0.0027398496], [0.0126409024, 0.002597802], [0.0120570752, 0.0024509331], [0.0114426013, 0.0022991996], [0.0107928272, 0.0021418111], [0.01010953, 0.0019795319], [0.00938643, 0.0018113315], [0.0086195966, 0.0016368399], [0.0078053713, 0.0014558806], [0.0069370416, 0.0012678023], [0.0060063621, 0.0010719318], [0.0050043175, 0.0008678946], [0.0039178578, 0.000655262], [0.0027266881, 0.0004337085], [0.0014069203, 0.0002058561]]


csv_file = open(parameters_file)
csv_reader = csv.reader(csv_file, delimiter=';')
line_count = 0
for row in csv_reader:
	if line_count == 0:
		line_count += 1
	else:
		#Read the parameters associated to rate R
		R = float(row[0])
		Claimed_Complexity = float(row[1])
		s = float(row[2])
		u = float(row[3])
		w = float(row[4])
		t = float(row[5])

		# Upper bound on the expected size of S
		bound_expected_size_S = max(s*H2((t-u)/s) + (1-s)*H2(u/(1-s)) - (1-R),0)
		
		# Asymptotic optimal parameters of Dumer91 decoder to decode a code of rate
		# (R-s)/(1-s) at relative distance u/(1-s)
		optimal_params = PARAMS_ISD_DUMER_91[line_count - 1]
		# Asymptotic complexity exponent of one call to the routine SOLVE-SUBPROBLEM (ISD Dumer91)
		cost_SUB_PROBLEM = (1-s)*isd_dumer91((R-s)/(1-s),u/(1-s), optimal_params[0]/(1-s), optimal_params[1]/(1-s))
			

		# Condition that must be verified in order for corrected RLPN additional steps to be negligible in front of the FFT
		assert(bound_expected_size_S + cost_SUB_PROBLEM <= s)
		line_count += 1

# Program Finish. All parameters met constraint.
