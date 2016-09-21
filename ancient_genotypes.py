import msprime as msp
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as st

class FreqError(Exception):
	pass

def ancient_sample(num_modern=1000,anc_time=200,Ne=10000,mu=1.25e-8,length=1000,num_rep=1000):
	samples = [msp.Sample(population=0,time=0)]*num_modern
	samples.extend([msp.Sample(population=0,time=anc_time)]*2)
	sims = msp.simulate(samples=samples,Ne=Ne,mutation_rate=mu,length=length,num_replicates=num_rep)
	freq = []
	anc = []
	for sim in sims:
		for position, variant in sim.variants():
			var_array = map(int,list(variant))
			cur_freq = sum(var_array[:-2])/float(num_modern)
			if cur_freq == 0 or cur_freq == 1: continue
			freq.append(cur_freq)
			anc.append(sum(var_array[-2:]))
	return np.array(freq), np.array(anc)

def ancient_sample_split(num_modern=1000,anc_time=200,split_time=400,Ne0=10000,Ne1=10000,mu=1.25e-8,length=1000,num_rep=1000):
	samples = [msp.Sample(population=0,time=0)]*num_modern
	samples.extend([msp.Sample(population=1,time=anc_time)]*2)
	pop_config = [msp.PopulationConfiguration(initial_size=Ne0),msp.PopulationConfiguration(initial_size=Ne1)]
	divergence = [msp.MassMigration(time=split_time,source=1,destination=0,proportion=1.0)]
	sims = msp.simulate(samples=samples,Ne=Ne0,population_configurations=pop_config,demographic_events=divergence,mutation_rate=mu,length=length,num_replicates=num_rep)
	freq = []
	anc = []
	for sim in sims:
		for position, variant in sim.variants():
			var_array = map(int,list(variant))
			cur_freq = sum(var_array[:-2])/float(num_modern)
			if cur_freq == 0 or cur_freq == 1: continue
			freq.append(cur_freq)
			anc.append(sum(var_array[-2:]))
	return np.array(freq), np.array(anc)


def ancient_sample_mix(num_modern=1000,anc_pop = 0, anc_time=200,mix_time=300,split_time=400,f=0.0,Ne0=10000,Ne1=10000,mu=1.25e-8,length=1000,num_rep=1000,coverage=False):
	if mix_time > split_time:
		print "mixture occurs more anciently than population split!"
		return None
	if f < 0 or f > 1:
		print "Admixture fraction is not in [0,1]"
		return None
	samples = [msp.Sample(population=0,time=0)]*num_modern
	samples.extend([msp.Sample(population=anc_pop,time=anc_time)]*2)
	pop_config = [msp.PopulationConfiguration(initial_size=Ne0),msp.PopulationConfiguration(initial_size=Ne1)]
	divergence = [msp.MassMigration(time=mix_time,source=0,destination=1,proportion = f),
			msp.MassMigration(time=split_time,source=1,destination=0,proportion=1.0)]
	sims = msp.simulate(samples=samples,Ne=Ne0,population_configurations=pop_config,demographic_events=divergence,mutation_rate=mu,length=length,num_replicates=num_rep)
	freq = []
	anc = []
	for sim in sims:
		for position, variant in sim.variants():
			var_array = map(int,list(variant))
			cur_freq = sum(var_array[:-2])/float(num_modern)
			if cur_freq == 0 or cur_freq == 1: continue
			freq.append(cur_freq)
			if not coverage:
				anc.append(sum(var_array[-2:]))
			else:
				num_reads = st.poisson.rvs(coverage)
				reads = np.random.choice(var_array[-2:],size=num_reads, replace=True)
				GL = st.binom.pmf(sum(reads),num_reads,[0,.5,1])
				GL = GL/sum(GL)
				print var_array[-2:]
				print reads
				print GL
				print GL/sum(GL)
				raw_input()
	return np.array(freq), np.array(anc)

def get_het_prob(freq,anc):
	anc_dict = {}
	for i in range(len(freq)):
		if freq[i] in anc_dict:
			anc_dict[freq[i]][anc[i]] += 1
		else:
			anc_dict[freq[i]] = [0.0,0.0,0.0]
			anc_dict[freq[i]][anc[i]] += 1
	unique_freqs = sorted(np.unique(freq))
	pHet = []
	for i in range(len(unique_freqs)):
		cur_anc = anc_dict[unique_freqs[i]]
		try:
			pHet.append(cur_anc[1]/(cur_anc[1]+cur_anc[2]))
		except ZeroDivisionError:
			pHet.append(None)
	return np.array(unique_freqs), np.array(pHet), anc_dict

def expected_het_anc(x0,t):
	return 1.0/(3.0/2.0+(2*x0-1)/(1+np.exp(2*t)-2*x0))

def expected_het_split(x0,t1,t2):
	return 1.0/(1.0/2.0+(np.exp(2*t1+t2))/(1+np.exp(2*t1)-2*x0))

def get_numbers_from_dict(anc_dict):
	het = []
	hom = []
	for freq in sorted(anc_dict.keys()):
		het.append(anc_dict[freq][1])
		hom.append(anc_dict[freq][2])
	return np.array(het), np.array(hom)

#freqs, het, hom should be FIXED, determiend by data
def het_hom_likelihood_anc(t,freqs,het,hom):
	#check if 0 or 1 in freqs
	if 0 in freqs or 1 in freqs:
		raise FreqError("Remove sites that are monomophric in modern population")
	pHetExpect = expected_het_anc(freqs,t)
	likeVec = het*np.log(pHetExpect)+hom*np.log(1-pHetExpect)
	return(sum(likeVec))
	
#freqs, het, hom should be FIXED, determiend by data
def het_hom_likelihood_split(t1,t2,freqs,het,hom):
	if 0 in freqs or 1 in freqs:
		raise FreqError("Remove sites that are monomophric in modern population")
	pHetExpect = expected_het_split(freqs,t1,t2)
	likeVec = het*np.log(pHetExpect)+hom*np.log(1-pHetExpect)
	return(sum(likeVec))

def het_hom_likelihood_mixture(t1,t2,t3,p,freqs,het,hom):
	if 0 in freqs or 1 in freqs:
		raise FreqError("Remove sites that are monomophric in modern population")
	pHetAnc = expected_het_anc(freqs,t1)
	pHetSplit = expected_het_split(freqs,t2,t3)
	pHetExpect = p*pHetAnc+(1.-p)*pHetSplit
	likeVec = het*np.log(pHetExpect)+hom*np.log(1-pHetExpect)
	return(sum(likeVec))

def chi_squared_anc(t,freqs,het,hom):
	pHetExpect = expected_het_anc(freqs,t)
	num = het+hom
	pHat = het/num
	pHat[np.isnan(pHat)]=0
	residuals = (np.sqrt(num)*(pHat-pHetExpect)**1/np.sqrt(pHetExpect*(1-pHetExpect)))
	return residuals	 

def cf_sum_anc(k,t,freqs,het,hom):
	num = het+hom
	pHetExpect = expected_het_anc(freqs,t)
	exp_sum = np.sum(1./(1-pHetExpect))
	exp_part = np.exp(1j*k*exp_sum)
	prod_internal = (1-pHetExpect+pHetExpect*np.exp(1j*k/(num*pHetExpect*(1-pHetExpect))))**num
	prod_internal[np.isnan(prod_internal)] = 1
	prod_part = np.prod(prod_internal)
	return exp_part*prod_part

def test_and_plot(anc_dict,x0Anc = st.uniform.rvs(size=1), x0Split = st.uniform.rvs(size=2),plot=True):
	het,hom = get_numbers_from_dict(anc_dict)
	freqs = np.sort(anc_dict.keys())
	ancTest = opt.fmin_l_bfgs_b(func=lambda x: -het_hom_likelihood_anc(x[0],freqs,het,hom), x0 = x0Anc, approx_grad=True,bounds=[[.0001,1000]],factr=10,pgtol=1e-15)
	splitTest = opt.fmin_l_bfgs_b(func=lambda x: -het_hom_likelihood_split(x[0],x[1],freqs,het,hom), x0 = x0Split, approx_grad=True,bounds=[[.0001,100],[.0001,100]],factr=10,pgtol=1e-15)
	if plot:
		tAnc = ancTest[0][0]
		t1 = splitTest[0][0]
		t2 = splitTest[0][1]
		hetAnc = expected_het_anc(freqs,ancTest[0][0])
		hetSplit = expected_het_split(freqs,splitTest[0][0],splitTest[0][1])
		plt.plot(freqs,het/(het+hom),'o',label="data")
		plt.plot(freqs,hetAnc,'r',label="anc, t = %f"%tAnc)
		plt.plot(freqs,hetSplit,'y',label="split, t1 = %f t2 = %f"%(t1,t2))
		plt.xlabel("Frequency")
		plt.ylabel("Proportion of het sites")
		plt.legend()
	return ancTest,splitTest
