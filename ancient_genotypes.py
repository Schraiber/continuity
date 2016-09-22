import msprime as msp
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.special as sp
import copy as cp
from scipy.sparse.linalg import expm_multiply as expma

class FreqError(Exception):
	pass

def ancient_sample_mix_multiple(num_modern=1000,anc_pop = 0, anc_num = 1, anc_time=200,mix_time=300,split_time=400,f=0.0,Ne0=10000,Ne1=10000,mu=1.25e-8,length=1000,num_rep=1000,coverage=False):
	if mix_time > split_time:
		print "mixture occurs more anciently than population split!"
		return None
	if f < 0 or f > 1:
		print "Admixture fraction is not in [0,1]"
		return None
	samples = [msp.Sample(population=0,time=0)]*num_modern
	samples.extend([msp.Sample(population=anc_pop,time=anc_time)]*(2*anc_num))
	pop_config = [msp.PopulationConfiguration(initial_size=Ne0),msp.PopulationConfiguration(initial_size=Ne1)]
	divergence = [msp.MassMigration(time=mix_time,source=0,destination=1,proportion = f),
			msp.MassMigration(time=split_time,source=1,destination=0,proportion=1.0)]
	sims = msp.simulate(samples=samples,Ne=Ne0,population_configurations=pop_config,demographic_events=divergence,mutation_rate=mu,length=length,num_replicates=num_rep)
	freq = []
	reads = []
	GT = []
	sim_num = 0
	for sim in sims:
		for position, variant in sim.variants():
			var_array = map(int,list(variant))
			cur_freq = sum(var_array[:-(2*anc_num)])/float(num_modern)
			if cur_freq == 0 or cur_freq == 1: continue
			freq.append(cur_freq)
			reads.append([])
			GT.append([])
			for i in range(anc_num):
				if i == 0: cur_GT = var_array[-2:]
				else: cur_GT = var_array[-(2*(i+1)):-(2*i)]
				cur_GT = sum(cur_GT)
				GT[-1].append(cur_GT)
				reads[-1].append([None,None])
				if coverage:
					num_reads = st.poisson.rvs(coverage)
					derived_reads = st.binom.rvs(num_reads, cur_GT/2.)
					reads[-1][-1] = (num_reads-derived_reads,derived_reads)
	return np.array(freq), GT, reads

def get_het_prob(freq,GT):
	anc_dict_list = []
	num_ind = len(GT[0])
	for ind in range(num_ind):
		anc_dict_list.append({})
	for i in range(len(freq)):
		for ind in range(num_ind):
			if freq[i] in anc_dict_list[ind]:
				anc_dict_list[ind][freq[i]][GT[i][ind]] += 1.0
			else:
				anc_dict_list[ind][freq[i]] = np.array([0.0,0.0,0.0])
				anc_dict_list[ind][freq[i]][GT[i][ind]] += 1.0
	unique_freqs = sorted(np.unique(freq))
	pHet = []
	for ind in range(num_ind):
		pHet.append([])
		for i in range(len(unique_freqs)):
			cur_anc = anc_dict_list[ind][unique_freqs[i]]
			try:
				pHet[-1].append(cur_anc[1]/(cur_anc[1]+cur_anc[2]))
			except ZeroDivisionError:
				pHet[-1].append(None)
	return np.array(unique_freqs), np.array(pHet), anc_dict_list


#read_dict is a list of arrays, sorted by freq
##the first level corresponds to the freqs in freq
##within each frequency, there are arrays of reads that can be passed to compute_read_like
def get_read_dict(freq,reads):
	read_dict = {}
	num_ind = len(reads[0])
	for i in range(len(freq)):
		if freq[i] in read_dict:
			read_dict[freq[i]].append(np.array(reads[i]))
		else:
			read_dict[freq[i]] = []
			read_dict[freq[i]].append(np.array(reads[i]))
	freqs = sorted(read_dict)
	read_list = []
	for freq in freqs:
		read_list.append(read_dict[freq])
	return freqs, read_list

def expected_het_anc(x0,t):
	return 1.0/(3.0/2.0+(2*x0-1)/(1+np.exp(2*t)-2*x0))

def expected_het_split(x0,t1,t2):
	return 1.0/(1.0/2.0+(np.exp(2*t1+t2))/(1+np.exp(2*t1)-2*x0))

def expected_moments_split(x0,t1,t2):
	Ehet = np.exp(-3.*t1-t2)*(1.+np.exp(2.*t1)-2.*x0)*x0
	Eder = 1./2.*np.exp(-3.*t1-t2)*(2.*x0+np.exp(2.*t1)*(2.*np.exp(t2)-1.)-1.)*x0
	Eanc = 1.0  - Ehet - Eder
	return Eanc, Ehet, Eder

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

#GLs should be a matrix
#freqs is the frequency of each site, should be same length as GLs
def GL_likelihood_split(t1,t2,freqs,GLs):
	expect = np.transpose(expected_moments_split(freqs,t1,t2))
	likePerLocus = np.sum(GLs*expect,axis=1)
	LL = np.sum(np.log(likePerLocus))
	return LL	

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

def test_and_plot_GL():
	return 0

def generate_genotypes(n):
	n -= 1 #NB: This is just because Python is dumb about what n means
	GTs = [[0],[1],[2]]
	for i in range(n):
		newGTs = []
		for GT in GTs:
			for j in (0,1,2):	
				newGT = cp.deepcopy(GT)
				newGT.append(j)
				newGTs.append(newGT)
		GTs = newGTs
	return np.array(GTs)

def generate_Q(n):
	Q = np.zeros((n,n))
	#NB: indexing is weird b/c Python. 
	#In 1-offset, Qii = -i*(i-1)/2, Qi,i-1 = i*(i-1)/2
	for i in range(1,n+1):
		Q[i-1,i-1] = -i*(i-1)/2
		Q[i-1,i-2] = i*(i-1)/2
	return Q

def generate_Qd(n):
	Q = np.zeros((n,n))
	for i in range(1,n+1):
		Q[i-1,i-1] = -i*(i+1)/2
		Q[i-1,i-2] = i*(i-1)/2
	return Q

#NB: Should be freq PER locus
def generate_x(freq,n):
	pows = range(1,n+1)
	xMat = np.array(map(lambda x: np.array(x)**pows,freq))
	return np.transpose(xMat)

def compute_Ey(freq,n,t1,t2):
	Qd = generate_Qd(n)
	Q = generate_Q(n)
	x = generate_x(freq,n)
	backward = expma(Qd*t1,x)
	Ey = np.vstack((np.ones(len(freq)),expma(Q*t2,backward)))
	return Ey

#NB: this does NOT include the combinatorial constant 
def compute_sampling_probs(Ey):
 	n = Ey.shape[0]-1 #NB: number of haploids, -1 because of the row of 1s at the top...
	numFreq = Ey.shape[1]
	probs = []
	for j in range(numFreq):
		probs.append([])
		for k in np.arange(n+1): #all possible freqs, including 0 and 1
		 	i = np.arange(0,n-k+1)
			cur_prob = (-1)**i*sp.binom(n-k,i)*Ey[i+k,j]
			cur_prob = np.sum(cur_prob)
			probs[-1].append(cur_prob)
	return np.array(probs)

def get_bounds_reads(reads):
	cur_max_a = 0
	cur_max_d = 0
	cur_min_a = np.inf
	cur_min_d = np.inf
	for i in range(len(reads)):
		for j in range(len(reads[i])):
			test_min_a = min(reads[i][j][:,0])
			test_max_a = max(reads[i][j][:,0])
			test_min_d = min(reads[i][j][:,1])
			test_max_d = max(reads[i][j][:,1])
			if test_min_a < cur_min_a: cur_min_a = test_min_a
			if test_max_a > cur_max_a: cur_max_a = test_max_a
			if test_min_d < cur_min_d: cur_min_d = test_min_d
			if test_max_d > cur_max_d: cur_max_d = test_max_d
	return cur_min_a, cur_max_a, cur_min_d, cur_max_d

def precompute_read_like(min_a,max_a,min_d,max_d):
	read_like = {}
	for a in range(min_a,max_a+1):
		for d in range(min_d,max_d+1):
			read_like[(a,d)] = st.binom.pmf(d,a+d,[0.,.5,1.])
	return read_like	

def precompute_all_read_like(reads):
	read_like = []
	for i in range(len(reads)):
		read_like.append([])
		for j in range(len(reads[i])):
			der = reads[i][j][:,1]
			total = reads[i][j][:,0]+reads[i][j][:,1]
			cur_like = np.vstack((st.binom.pmf(der,total,0.),st.binom.pmf(der,total,.5),st.binom.pmf(der,total,1.)))
			read_like[-1].append(np.transpose(cur_like))
	return read_like

#NB: Takes the WHOLE matrix of genotypes
#reads is an array where each row is a sample, reads[:,0] is ancestral reads, reads[:,1] is derived reads at that site
def compute_read_like(reads,GTs):
	p = np.array([0,.5,1])
	read_like = []
	#Hack to deal with the fact that some freqs may not have multiple sites...
	try:
		der = reads[:,1]
		total = reads[:,0]+reads[:,1]
	except IndexError:
		der = reads[1]
		total = reads[0]+reads[1]
	for GT in GTs:
		cur_likes = np.product(st.binom.pmf(der,total,p[GT]))
		read_like.append(cur_likes)
	return np.array(read_like)

#NB: Sampling prob is just for ONE frequency in this case
def compute_genotype_sampling_probs(sampling_prob, GTs):
	GT_prob = map(lambda GT: 2**sum(GT==1)*sampling_prob[sum(GT)],GTs)
	return np.array(GT_prob)

#reads is a list of arrays, sorted by freq
##the first level corresponds to the freqs in freq
##within each frequency, there are arrays of reads that can be passed to compute_read_like
def compute_GT_like(reads,freq,t1,t2,detail=False):
	if reads[0][0].ndim == 1:
		n_diploid = 1
	else:
		n_diploid = len(reads[0][0])
	n_haploid = 2*n_diploid
	GTs = generate_genotypes(n_diploid)	
	Ey = compute_Ey(freq,n_haploid,t1,t2)
	sampling_prob = compute_sampling_probs(Ey)
	per_site_like = []
	for i in range(len(freq)):
		GT_prob = compute_genotype_sampling_probs(sampling_prob[i,:],GTs)
		for j in range(len(reads[i])):
			read_like = compute_read_like(reads[i][j],GTs)
			cur_prob = sum(read_like*GT_prob)
			per_site_like.append(cur_prob)	
	LL = np.log(per_site_like)
	if detail: print t1, t2, -sum(LL)
	return LL

#This expects a precomputed dictionary of genotype likelihoods that are observed in the data, using precompute_read_like
#TODO: If implementing with error, will need to move the precompute stage *inside* the likelihood. Should still be better than old way
def compute_GT_like_precompute_dict(reads,freq,t1,t2,read_like,detail=False):
	if reads[0][0].ndim == 1:
		n_diploid = 1
	else:
		n_diploid = len(reads[0][0])
	n_haploid = 2*n_diploid
	GTs = generate_genotypes(n_diploid)	
	Ey = compute_Ey(freq,n_haploid,t1,t2)
	sampling_prob = compute_sampling_probs(Ey)
	per_site_like = []
	for i in range(len(freq)):
		GT_prob = compute_genotype_sampling_probs(sampling_prob[i,:],GTs)
		for j in range(len(reads[i])):
			cur_like = []
			for GT in GTs:
				cur_like.append(1.0)
				for ind in range(len(GT)):
					cur_like[-1] *= read_like[(reads[i][j][ind][0],reads[i][j][ind][1])][GT[ind]]
			cur_prob = sum(np.array(cur_like)*GT_prob)
			per_site_like.append(cur_prob)	
	LL = np.log(per_site_like)
	if detail: print t1, t2, -sum(LL)
	return LL

#this expects reads to actually be an array of GLs, not an array of reads
##same strucutre as the default version without precomputing
#this is probably not that useful
#TODO:This is broken 
def compute_GT_like_precompute(reads,freq,t1,t2,detail=False):
	if reads[0][0].ndim == 1:
		n_diploid = 1
	else:
		n_diploid = len(reads[0][0])
	n_haploid = 2*n_diploid
	good_range = np.arange(0,n_diploid+1)
	GTs = generate_genotypes(n_diploid)	
	Ey = compute_Ey(freq,n_haploid,t1,t2)
	sampling_prob = compute_sampling_probs(Ey)
	per_site_like = []
	for i in range(len(freq)):
		GT_prob = compute_genotype_sampling_probs(sampling_prob[i,:],GTs)
		for j in range(len(reads[i])):
			cur_like = []
			print reads[i][j]
			for GT in GTs:
				print GT
				print reads[i][j][good_range,np.array(GT)]
				cur_like.append(np.product(reads[i][j][good_range,np.array(GT)]))
			print reads[i][j]
			print GTs
			print cur_like
			raw_input()
			cur_prob = sum(cur_like*GT_prob)
			per_site_like.append(cur_prob)	
	LL = np.log(per_site_like)
	if detail: print t1, t2, -sum(LL)
	return ll
