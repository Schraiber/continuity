import msprime as msp
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.special as sp
import copy as cp
from scipy.sparse.linalg import expm_multiply as expma
import itertools
from copy import deepcopy
from numpy import random as rn
import sklearn.cluster as cl
from joblib import Parallel, delayed
import pandas

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
		for variant in sim.variants():
			var_array = variant.genotypes
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


#NB: This function does not support admixture
#NB: This function models a very simple structure where all pops are splits off of a long ancestral lineage
def ancient_sample_many_pops(num_modern=1000,anc_pop = [0], anc_per_pop = [1], anc_time=[200],split_time=[400],Ne0=10000,NeAnc=[10000],mu=1.25e-8,length=1000,num_rep=1000,coverage=False, error=None, seed=None):
	if not (len(anc_pop) == len(anc_per_pop) == len(anc_time) == len(split_time) == len(NeAnc)):
		print "There are an unequal number of elements in the vectors specifying the ancient samples"
		print "len(anc_pop) = %d, len(anc_per_pop) = %d, len(anc_time) = %d, len(split_time) = %d, len(NeAnc) = %d"%(len(anc_pop),len(anc_per_pop),len(anc_time),len(split_time),len(NeAnc))
		return None
	#make errors:
	if error is None:
		error = np.zeros(sum(anc_per_pop))
	#make modern samples
	samples = [msp.Sample(population=0,time=0)]*num_modern
	anc_num = 0
	num_pop = max(anc_pop)+1
	split_times = [0]*num_pop
	Ne = [0]*num_pop
	Ne[0] = Ne0
	label = []
	for i in range(len(anc_pop)):
		if anc_time[i] > split_time[i] and anc_pop[i] != 0:
			print "The sample is more ancient than the population it belongs to"
			print anc_time[i], split_time[i]
			return None
		samples.extend([msp.Sample(population=anc_pop[i],time=anc_time[i])]*(2*anc_per_pop[i]))
		label.extend([i]*anc_per_pop[i])
		cur_pop = anc_pop[i]
		anc_num += anc_per_pop[i]
		if cur_pop == 0: continue #Hack to just avoid fucking up things for dudes in the first pop
		split_times[cur_pop] = split_time[i]
		Ne[cur_pop] = NeAnc[i]	
	pop_config = [msp.PopulationConfiguration(initial_size=Ne[0])]
	divergence = []
	for pop in range(1,len(split_times)):
		pop_config.append(msp.PopulationConfiguration(initial_size=Ne[pop]))
		divergence.append(msp.MassMigration(time=split_times[pop],source=pop,destination=0,proportion=1.0))
	sims = msp.simulate(samples=samples,Ne=Ne[0],population_configurations=pop_config,demographic_events=divergence,mutation_rate=mu,length=length,num_replicates=num_rep,random_seed=seed)
	freq = []
	read_dicts = [{} for i in range(num_pop)]
	GT = []
	reads = []
	for ind in range(anc_num):
		GT.append([])
		reads.append([])
	sim_num = 0
	for sim in sims:
		for variant in sim.variants():
			var_array = variant.genotypes
			cur_freq = sum(var_array[:-(2*anc_num)])/float(num_modern)
			if cur_freq == 0 or cur_freq == 1: continue
			freq.append(cur_freq)
			reads_per_pop = [[] for i in range(num_pop)]
			for i in range(anc_num):
				ind_num = anc_num-i-1 #NB: indexing to get the output vector to be in the right order
				if i == 0: cur_GT = var_array[-2:]
				else: cur_GT = var_array[-(2*(i+1)):-(2*i)]
				cur_GT = sum(cur_GT)
				GT[ind_num].append(cur_GT)
				cur_pop = label[i]
				reads_per_pop[cur_pop].append([None,None])
				reads[ind_num].append([None,None])
				if coverage:
					num_reads = st.poisson.rvs(coverage)
					#num_reads = st.geom.rvs(1./coverage)
					p_der = cur_GT/2.*(1-error[ind_num])+(1-cur_GT/2.)*error[ind_num]
					derived_reads = st.binom.rvs(num_reads, p_der)
					reads[ind_num][-1] = (num_reads-derived_reads,derived_reads)
					reads_per_pop[cur_pop][-1] = [num_reads-derived_reads,derived_reads]
			for i in range(num_pop):
				if cur_freq in read_dicts[i]:
					read_dicts[i][cur_freq].append(np.array(reads_per_pop[i]))
				else:
					read_dicts[i][cur_freq] = [np.array(reads_per_pop[i])]
	freqs = sorted(read_dicts[0])
	read_lists = []
	for i in range(len(read_dicts)):
		read_lists.append([])
		for curfreq in freqs:
			read_lists[-1].append(np.array(read_dicts[i][curfreq]))
	return np.array(freq), GT, reads, read_lists, np.array(freqs)	

#Writes a beagle genotype likelihood file for data from the simulations
#NB: simulates modern individuals from the allele frequencies and HW equilibrium
#NB: modern individuals just have a genotype likelihood of 1 for the true genotype
def write_beagle_output(freq, reads, file_name, num_modern = 100):
	outfile = open(file_name,"w")
	outfile.write("marker\tallele1\tallele2\t")
	modern = ['\t'.join([''.join(("modern",str(i)))]*3) for i in range(num_modern)]
	anc = ['\t'.join([''.join(("ancient",str(i)))]*3) for i in range(len(reads))]
	outfile.write("%s\t%s\n"%('\t'.join(modern),'\t'.join(anc)))
	for i in range(len(freq)):
		HW = ((1-freq[i])**2, 2*freq[i]*(1.-freq[i]),freq[i]**2)
		modern_genotypes_draw = rn.multinomial(1,HW,num_modern)
		modern_genotypes = map(lambda x: np.where(x==1)[0][0],modern_genotypes_draw)
		modern_GL = np.zeros((num_modern,3))
		modern_GL[range(num_modern),modern_genotypes] = 1
		outfile.write("marker_%d\t0\t1"%i)
		for ind in modern_GL:
			outfile.write("\t")
			outfile.write("\t".join(map(str,ind)))				
		for ind in reads:
			cur_GL  = st.binom.pmf(ind[i][1],sum(ind[i]),[0,.5,1])
			outfile.write("\t")
			outfile.write("\t".join(map(str,cur_GL/sum(cur_GL))))
		outfile.write("\n")			
	outfile.close()

def get_inds_pops(reads_file,ind_file):
	reads = open(reads_file)
	header = reads.readline()
	reads.close()
	headerSplit = header.strip().split()
        inds_alleles = np.array(headerSplit[3:])
        der_indices = np.arange(len(inds_alleles),step=3)
        anc_indices = np.arange(1,len(inds_alleles),step=3)
        inds = [x.split("_")[0] for x in inds_alleles[der_indices]]
	inds_pops = pandas.read_table(ind_file,delim_whitespace=True,header=None)
	unique_pops = np.unique(inds_pops[ [ind in inds for ind in inds_pops[0]] ][[2]])
	label_names = [inds_pops[inds_pops[0] == ind][2].values for ind in inds]
	label = [int(np.where(name == unique_pops)[0]) for name in label_names]
	pops = [list(np.where(np.array(label)==i)[0]) for i in range(len(unique_pops))]
	return unique_pops, label_names, label, pops, inds

def parse_reads(read_file_name,cutoff=0):
	read_file = open(read_file_name)
	freq = []
	samples = []
	header = read_file.readline()
	headerSplit = header.strip().split()
	inds_alleles = np.array(headerSplit[3:])
	der_indices = np.arange(len(inds_alleles),step=3)
	anc_indices = np.arange(1,len(inds_alleles),step=3)
	inds = [x.split("_")[0] for x in inds_alleles[der_indices]]
	reads = [[] for i in range(len(inds))]
	for line_no, line in enumerate(read_file):
		if line_no % 10000 == 0: print line_no
		splitLine = line.strip().split()
		read_counts = np.array(map(int,splitLine[3:]))
		der_counts = read_counts[der_indices]
		anc_counts = read_counts[anc_indices]
		sample_has_reads = (der_counts > 0) | (anc_counts > 0)
		samples_with_reads = sum(sample_has_reads)
		if float(samples_with_reads)/len(inds) < cutoff: continue
		freq.append(float(splitLine[2]))
		for i in range(len(inds)):
			reads[i].append((anc_counts[i],der_counts[i]))
	return np.array(freq), reads, inds

#pops is a list of lists of inds to put into pops
#e.g. [[0,3],[1,2]] says put inds 0 and 3 in pop 1, inds 1 and 2 in pop 2 
def parse_reads_by_pop(read_file_name,ind_file,cutoff=0):
	read_file = open(read_file_name)
	header = read_file.readline()
	headerSplit = header.strip().split()
	inds_alleles = np.array(headerSplit[3:]) #TODO: Make all these have variable start indices so that it can handle k, n
	der_indices = np.arange(len(inds_alleles),step=3)
	anc_indices = np.arange(1,len(inds_alleles),step=3)
	inds = [x.split("_")[0] for x in inds_alleles[der_indices]]
	inds_pops = pandas.read_table(ind_file,delim_whitespace=True,header=None)
	unique_pops = np.unique(inds_pops[ [ind in inds for ind in inds_pops[0]] ][[2]])
	label_names = [inds_pops[inds_pops[0] == ind][2].values for ind in inds]
	label = [int(np.where(name == unique_pops)[0]) for name in label_names]
	pops = [list(np.where(np.array(label)==i)[0]) for i in range(len(unique_pops))]
	read_dicts = [{} for i in range(max(label)+1)]
	for line_no, line in enumerate(read_file):
		if line_no % 10000 == 0: print line_no
		splitLine = line.strip().split()
		read_counts = np.array(map(int,splitLine[3:])) #TODO: Make this so it can handle either counts or k, n by making the first index a variable
		der_counts = read_counts[der_indices]
		anc_counts = read_counts[anc_indices]
		sample_has_reads = (der_counts > 0) | (anc_counts > 0)
		samples_with_reads = sum(sample_has_reads)
		if float(samples_with_reads)/len(inds) < cutoff: continue
		cur_freq = float(splitLine[2]) #TODO: Generalize to n, k
		if cur_freq == 0 or cur_freq == 1: 
			print "Ignoring alleles that are frequency 0 or frequency 1 in reference population"
			continue
		reads_per_pop = [[] for i in range(max(label)+1)]
		for i in range(len(inds)):
			cur_pop = label[i]
			cur_counts = (anc_counts[i], der_counts[i])
			reads_per_pop[cur_pop].append(cur_counts)
		for i in range(max(label)+1):
			if cur_freq in read_dicts[i]:
				read_dicts[i][cur_freq].append(np.array(reads_per_pop[i]))
			else:
				read_dicts[i][cur_freq] = [np.array(reads_per_pop[i])]
	freqs = sorted(read_dicts[0])
	read_lists = []
	for i in range(len(read_dicts)):
		read_lists.append([])
		for freq in freqs:
			read_lists[-1].append(np.array(read_dicts[i][freq]))
	return unique_pops, label_names, label, pops, inds, freqs, read_lists


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

#pops is a list of lists of inds to put into pops
#e.g. [[0,3],[1,2]] says put inds 0 and 3 in pop 1, inds 1 and 2 in pop 2 
def make_read_dict_by_pop(freq,reads,pops):
	num_ind = len(reads)
	num_pops = len(pops)
	num_ind_in_pops = sum(map(len,pops))
	if num_ind_in_pops != num_ind:
		print "Number of inds to cluster into pops is different from number of inds in reads"
		print "Number in reads: %d, number in pops: %d, allocated as %s"%(num_ind,num_ind_in_pops,str(pops))
	read_dicts = []
	for pop in pops:
		cur_num_ind = len(pop)
		cur_dict = {}
		for i in range(len(freq)):
			cur_reads = []
			for ind in pop:
				cur_reads.append(reads[ind][i])
			if freq[i] in cur_dict:
				cur_dict[freq[i]].append(np.array(cur_reads))
			else:
				cur_dict[freq[i]] = []
				cur_dict[freq[i]].append(np.array(cur_reads))
		read_dicts.append(cur_dict)
	freqs = sorted(read_dicts[0])
	read_lists = []
	for i in range(len(read_dicts)):
		read_lists.append([])
		for freq in freqs:
			read_lists[-1].append(np.array(read_dicts[i][freq]))
	return freqs, read_lists


#NB: includes k = 0 case
def generate_Qd_het(n):
	Q = np.zeros((n+1,n+1))
	for k in range(n+1):
		Q[k,k] = -k*(n-k+1)
		if k > 0: Q[k,k-1] = .5*k*(k-1)
		if k < n: Q[k,k+1] = .5*(k-n-1)*(k-n)
	return Q

def generate_Q_het(n):
	Q = np.zeros((n+1,n+1))
	for k in range(n+1):
		Q[k,k] = -k*(n-k)
		if k > 0: Q[k,k-1] = .5*k*(k-1)
		if k < n: Q[k,k+1] = .5*(n-k-1)*(n-k)
	return Q

def generate_het(freq, n):
	pows = np.arange(0,n+1)
	hetMat = np.array(map(lambda x: x**pows*(1-x)**(n-pows),np.array(freq)))
	return np.transpose(hetMat)

def compute_Ehet(freq, n, t1, t2):
	Qd = generate_Qd_het(n)
	Q = generate_Q_het(n)
	het = generate_het(freq,n)
	backward = expma(Qd*t1,het)
	Ehet = expma(Q*t2,backward)
	return np.transpose(Ehet)	

def get_bounds_reads(reads):
	reads_array = np.array(reads)
	min_a = np.min(reads_array[:,:,0])
	max_a = np.max(reads_array[:,:,0])
	min_d = np.min(reads_array[:,:,1])
	max_d = np.max(reads_array[:,:,1])
	return min_a, max_a, min_d, max_d

def precompute_read_like(min_a,max_a,min_d,max_d):
	read_like = np.zeros((max_a-min_a+1,max_d-min_d+1,3))
	for a in range(min_a,max_a+1):
		for d in range(min_d,max_d+1):		
			read_like[a-min_a,d-min_d,:] = st.binom.pmf(d,a+d,[0,.5,1])
	return read_like

#error is a vector of error rates, one for each sample
#probably can be optmized by having a different min and max for each ind...
def precompute_read_like_error(min_a,max_a,min_d,max_d,errors):
	num_ind = len(errors)
	read_like = np.zeros((max_a-min_a+1,max_d-min_d+1,num_ind,3))
	p = np.array([errors,errors/2+(1-errors)/2,1-errors])
	a = np.arange(min_a,max_a+1)
	d = np.arange(min_d,max_d+1)
	for i in range(num_ind):
		for G in range(3):
			read_like[:,:,i,G] = st.binom.pmf(d,a[:,None]+d,p[G,i])
	#read_like[a-min_a,d-min_d,:,:] = np.transpose(st.binom.pmf(d,a[:,None]+d,p))
	#for a in range(min_a,max_a+1):
	#	for d in range(min_d,max_d+1):
	#		read_like[a-min_a,d-min_d,:,:] = np.transpose(st.binom.pmf(d,a+d,p))
	return read_like
		

#expects reads to be in the format with all individuals in a single population
def bound_and_precompute_read_like(reads):
	min_a,max_a,min_d,max_d = get_bounds_reads(reads)
	read_like = precompute_read_like(min_a,max_a,min_d,max_d)
	return min_a, min_d, read_like


def compute_all_read_like(reads,precompute_like,min_a,min_d):
	read_likes = precompute_like[reads[:,:,0]-min_a,reads[:,:,1]-min_d,:]
	return read_likes

def compute_all_read_like_error(reads,precompute_like,min_a,min_d):
	ind_nums = np.arange(len(reads[0]))
	read_likes = precompute_like[reads[:,ind_nums,0]-min_a,reads[:,ind_nums,1]-min_d,ind_nums,:]
	return read_likes

#NB: This expects the read likelihoods, 
#read_likes[i][j][k] = probability of the reads at the ith site for the jth individual assuming genotype k
def read_prob_DP(read_likes):
	num_sites = len(read_likes)
	num_ind = len(read_likes[0])
	z = np.zeros((num_sites,num_ind,2*num_ind+1)) #the +1 is because you could have an allele freq of 2n
	#initialize
	z[:,0,:3] = np.array([1,2,1])*read_likes[:,0,:]
	#loop
	for j in range(1,num_ind):
		for k in range(0,(j+1)*2+1):
			z[:,j,k] = read_likes[:,j,0]*z[:,j-1,k]+2*read_likes[:,j,1]*z[:,j-1,k-1]+read_likes[:,j,2]*z[:,j-1,k-2]
	h = z[:,num_ind-1,:]	
	return h


def optimize_single_pop_thread(r, freqs, min_a, max_a, min_d, max_d, detail = False, continuity=False):
	if continuity:
		t_bounds = np.array((1e-10,10))
	else:
		t_bounds = np.array(((1e-10,10),(1e-10,10)))
	num_ind_in_pop = len(r[0][0])
	params_init = np.hstack((st.uniform.rvs(size=2),st.uniform.rvs(size=num_ind_in_pop,scale=.1)))
	e_bounds = np.transpose(np.vstack((np.full(num_ind_in_pop,1e-10),np.full(num_ind_in_pop,.2))))
	bounds = np.vstack((t_bounds,e_bounds))	
	if continuity:
		params_init = np.delete(params_init, 1)
		cur_opt = opt.fmin_l_bfgs_b(func = lambda x: -sum(likelihood_error(r,freqs,x[0],0,x[1:],min_a,max_a,min_d,max_d,detail=detail)), x0 = params_init, approx_grad = True, bounds = bounds)#, factr = 1, pgtol = 1e-15)
	else:
		cur_opt = opt.fmin_l_bfgs_b(func = lambda x: -sum(likelihood_error(r,freqs,x[0],x[1],x[2:],min_a,max_a,min_d,max_d,detail=detail)), x0 = params_init, approx_grad = True, bounds = bounds)#, factr = 1, pgtol = 1e-15)
	print cur_opt[0], cur_opt[1]
	return cur_opt


def optimize_pop_params_error_parallel(freq,reads,pops,num_core = 1, detail=False, continuity=False):
	min_a, max_a, min_d, max_d = get_bounds_reads(reads)
	freqs, read_lists = make_read_dict_by_pop(freq,reads,pops)
	opts = Parallel(n_jobs=num_core)(delayed(optimize_single_pop_thread)(r, freqs, min_a, max_a, min_d, max_d, detail, continuity) for r in read_lists)
	return opts

def optimize_params_one_pop(freq,reads,pop,detail=False):
	all_inds = set(range(len(reads)))
	not_in_pop = all_inds.difference(pop)
	freqs, read_lists = make_read_dict_by_pop(freq,reads,[pop,list(not_in_pop)])
	min_a, min_d, read_like = bound_and_precompute_read_like(reads)
	cur_opt = opt.fmin_l_bfgs_b(func=lambda x: -sum(compute_GT_like_DP(read_lists[0],freqs,x[0],x[1],read_like,min_a,min_d,detail=detail)), x0 = st.uniform.rvs(size=2), approx_grad=True,bounds=[[.00001,10],[.00001,10]],epsilon=.001)#, factr=10, pgtol=1e-10) 
	return cur_opt
	

def compute_GT_like_DP_error(reads,freq,t1,t2,precompute_read_prob,min_a,min_d,detail=False):
	if reads[0][0].ndim == 1:
		n_diploid = 1
	else:
		n_diploid = len(reads[0][0])
	n_haploid = 2*n_diploid
	sampling_prob = compute_Ehet(freq,n_haploid,t1,t2)
	like_per_freq = []
	for i in range(len(freq)):
		read_prob_per_site = compute_all_read_like_error(reads[i],precompute_read_prob,min_a,min_d)
		read_prob = read_prob_DP(read_prob_per_site)
		like_per_freq.append(sum(np.log(np.dot(read_prob,sampling_prob[i]))))
	if detail: print t1, t2, -sum(like_per_freq)
	return like_per_freq

def likelihood_error(reads,freq,t1,t2,error,min_a,max_a,min_d,max_d,detail=False):
	if t1 < 0 or t2 < 0 or np.any(error<0) or np.any(error>1): 
		if detail: print t1, t2, 1e300
		return -1e300
	read_probs = precompute_read_like_error(min_a,max_a,min_d,max_d,error)
	if detail > 1: print error
	return compute_GT_like_DP_error(reads,freq,t1,t2,read_probs,min_a,min_d,detail=detail)

