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
def ancient_sample_many_pops(num_modern=1000,anc_pop = [0], anc_per_pop = [1], anc_time=[200],split_time=[400],Ne0=10000,NeAnc=[10000],mu=1.25e-8,length=1000,num_rep=1000,coverage=False):
	if not (len(anc_pop) == len(anc_per_pop) == len(anc_time) == len(split_time) == len(NeAnc)):
		print "There are an unequal number of elements in the vectors specifying the ancient samples"
		print "len(anc_pop) = %d, len(anc_per_pop) = %d, len(anc_time) = %d, len(split_time) = %d, len(NeAnc) = %d"%(len(anc_pop),len(anc_per_pop),len(anc_time),len(split_time),len(NeAnc))
		return None
	#make modern samples
	samples = [msp.Sample(population=0,time=0)]*num_modern
	anc_num = 0
	num_pop = max(anc_pop)+1
	split_times = [0]*num_pop
	Ne = [0]*num_pop
	Ne[0] = Ne0
	for i in range(len(anc_pop)):
		if anc_time[i] > split_time[i] and anc_pop[i] != 0:
			print "The sample is more ancient than the population it belongs to"
			print anc_time[i], split_time[i]
			return None
		samples.extend([msp.Sample(population=anc_pop[i],time=anc_time[i])]*(2*anc_per_pop[i]))
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
	sims = msp.simulate(samples=samples,Ne=Ne[0],population_configurations=pop_config,demographic_events=divergence,mutation_rate=mu,length=length,num_replicates=num_rep)
	freq = []
	reads = []
	GT = []
	for ind in range(anc_num):
		reads.append([])
		GT.append([])
	sim_num = 0
	for sim in sims:
		for variant in sim.variants():
			var_array = variant.genotypes
			cur_freq = sum(var_array[:-(2*anc_num)])/float(num_modern)
			if cur_freq == 0 or cur_freq == 1: continue
			freq.append(cur_freq)
			for i in range(anc_num):
				ind_num = anc_num-i-1 #NB: indexing to get the output vector to be in the right order
				if i == 0: cur_GT = var_array[-2:]
				else: cur_GT = var_array[-(2*(i+1)):-(2*i)]
				cur_GT = sum(cur_GT)
				GT[ind_num].append(cur_GT)
				reads[ind_num].append([None,None])
				if coverage:
					num_reads = st.poisson.rvs(coverage)
					derived_reads = st.binom.rvs(num_reads, cur_GT/2.)
					reads[ind_num][-1] = (num_reads-derived_reads,derived_reads)
	return np.array(freq), GT, reads

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
	for line in read_file:
		splitLine = line.strip().split()
		read_counts = np.array(map(int,splitLine[3:]))
		der_counts = read_counts[der_indices]
		anc_counts = read_counts[anc_indices]
		sample_has_reads = (der_counts > 0) | (anc_counts > 0)
		samples_with_reads = sum(sample_has_reads)
		if float(samples_with_reads)/len(inds) < cutoff: continue
		freq.append(float(splitLine[2]))
		for i in range(len(inds)):
			reads[i].append((der_counts[i],anc_counts[i]))
	return np.array(freq), reads, inds
	

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

def get_het_prob_old(freq,GT):
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

def get_het_prob(freq,GT):
	anc_dict_list = []
	num_ind = len(GT)
	for ind in range(num_ind):
		anc_dict_list.append({})
	for i in range(len(freq)):
		for ind in range(num_ind):
			if freq[i] in anc_dict_list[ind]:
				anc_dict_list[ind][freq[i]][GT[ind][i]] += 1.0
			else:
				anc_dict_list[ind][freq[i]] = np.array([0.0,0.0,0.0])
				anc_dict_list[ind][freq[i]][GT[ind][i]] += 1.0
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



def expected_het_anc(x0,t):
	return 1.0/(3.0/2.0+(2*x0-1)/(1+np.exp(2*t)-2*x0))

def expected_het_split(x0,t1,t2):
	return 1.0/(1.0/2.0+(np.exp(2*t1+t2))/(1+np.exp(2*t1)-2*x0))

def expected_moments_split(x0,t1,t2):
	Ehet = np.exp(-3.*t1-t2)*(1.+np.exp(2.*t1)-2.*x0)*x0
	Eder = 1./2.*np.exp(-3.*t1-t2)*(2.*x0+np.exp(2.*t1)*(2.*np.exp(t2)-1.)-1.)*x0
	Eanc = 1.0  - Ehet - Eder
	return Eanc, Ehet, Eder

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

def test_and_plot(anc_dict,x0Anc = st.uniform.rvs(size=1), x0Split = st.uniform.rvs(size=2),plot=True,title=""):
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
		plt.plot(freqs,hetSplit,'y',label="split, t1 = %0.2f t2 = %0.2f"%(t1,t2))
		plt.xlabel("Frequency")
		plt.ylabel("Proportion of het sites")
		plt.legend()
		plt.title(title)
	return ancTest,splitTest

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
	read_like = np.zeros((num_ind,max_a-min_a+1,max_d-min_d+1,3))
	p = np.array([errors,errors/2+(1-errors)/2,1-errors])
	for a in range(min_a,max_a+1):
		for d in range(min_d,max_d+1):
			read_like[:,a-min_a,d-min_d,:] = np.transpose(st.binom.pmf(d,a+d,p))
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
	read_likes = precompute_like[:,reads[:,:,0]-min_a,reads[:,:,1]-min_d,:]
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

############################################################################
###########PARTITION########################################################
#CODE IS FROM http://jeromekelleher.net/generating-integer-partitions.html
def partition(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

#CODE IS FROM http://stackoverflow.com/a/19369410
def slice_by_lengths(lengths, the_list):
    for length in lengths:
        new = []
        for i in range(length):
            new.append(the_list.pop(0))
        yield new

def partitions(my_list):
    partitions = partition(len(my_list))
    permed = []
    for each_partition in partitions:
        permed.append(set(itertools.permutations(each_partition, len(each_partition))))

    for each_tuple in itertools.chain(*permed):
        yield list(slice_by_lengths(each_tuple, deepcopy(my_list)))
#########################################################################
########################################################################

#This wasnts the "raw" reads data
def find_best_config(freq,reads,detail=False):
	num_ind = len(reads)
	pars = []
	lnL = []
	parts = []
	for partition in partitions(range(num_ind)):
		print "Processing %s"%partition
		cur_opt = optimize_pop_params(freq,reads,partition,detail=detail)	
		cur_lnL = sum(map(lambda x: x[1],cur_opt))
		cur_pars = map(lambda x: x[0], cur_opt)
		parts.append(partition)
		lnL.append(cur_lnL)
		pars.append(cur_pars)
		print 2*(2*len(partition))+cur_lnL
	return parts, lnL, pars

#TODO: This does not guarantee that the number of clusters remains at k
#TODO: Probably need to implement a proper EM algorithm
#TODO: Proper EM might be hard. 
#TODO: Maybe just hack so that if one cluster gets empty, you pop a dude out?
def cluster_anc(freq,reads,k,num_iter=10, detail=False):
	num_ind = len(reads)
	all_separate_pops = []
	for i in range(num_ind):
		all_separate_pops.append([i])
	opts_separate = optimize_pop_params(freq,reads,all_separate_pops,detail=detail)
	lnL_separate = map(lambda x: x[1],opts_separate)
	freqs, reads_per_ind = make_read_dict_by_pop(freq,reads,all_separate_pops)
	min_a, min_d, read_prob = bound_and_precompute_read_like(reads)
	first_inds = rn.choice(num_ind,k)
	params = [o[0] for o in np.array(opts_separate)[first_inds]]
	pop_labels = np.zeros(num_ind)
	for i in range(num_iter):
		indLnLBest = []
		for j in range(num_ind):
			indLnL = np.full(k,-np.inf)
			for l in range(k):
				indLnL[l] = sum(compute_GT_like_DP(reads_per_ind[j],freqs, params[l][0],params[l][1],read_prob,min_a,min_d,detail=False))
			pop_labels[j] = np.argmax(indLnL)
			indLnLBest.append(-indLnL[pop_labels[j]])
		new_pops = np.array([np.where(pop_labels==i)[0].tolist() for i in range(k)])
		#this should make sure that every pop has a dude in it
		for p,pop in enumerate(new_pops):
			if np.array_equal(pop,[]):
				new_guy = np.argmax(np.array(indLnLBest)-np.array(lnL_separate))
				new_pops[pop_labels[new_guy]].remove(new_guy)
				new_pops[p] = [new_guy]
				indLnLBest[new_guy] = lnL_separate[new_guy]
		#if i > 1 and new_pops == pops: break
		pops = new_pops
		print pop_labels, pops
		opts = optimize_pop_params(freq,reads,pops,detail=detail)
		params = [o[0] if o else None for o in opts]
	return opts, pops

def chunk(num_ind,k):
	seq = range(num_ind)
	rn.shuffle(seq)
	avg = num_ind/float(k)
	out = []
	last = 0.0
	while last < num_ind:
		out.append(sorted(seq[int(last):int(last+avg)]))
		last += avg
	return out

def cluster_k(freq, reads, k, num_iter=10, initialize = "random", detail=False):
	num_ind = len(reads)
	if initialize is "random":
		first_inds = rn.choice(num_ind,k)
		cur_pops = chunk(num_ind,k)
	elif initialize is "kmeans":
		all_separate = [[i] for i in range(num_ind)]
		sep_opts = optimize_pop_params(freq,reads,all_separate,detail=detail)
		pars = np.array(map(lambda x: x[0], sep_opts))
		kmeans = cl.KMeans(n_clusters=k).fit(pars)
		labels = kmeans.labels_
		cur_pops = [[] for i in range(k)]
		for i in range(len(labels)):
			cur_pops[labels[i]].append(i)
				
	else:
		print "Unknown initialization procedure"
		return 0
	cur_opts = optimize_pop_params(freq,reads,cur_pops,detail=detail)
	calculated = {}
	for i in range(len(cur_pops)):
		calculated[tuple(cur_pops[i])] = cur_opts[i]
	print cur_pops, sum(map(lambda x: x[1], cur_opts))
	for i in range(num_iter):
		for ind in range(num_ind):
			best_lamb = 0
			best_pop = []
			old_pop = map(lambda x: ind in x, cur_pops).index(True)
			if len(cur_pops[old_pop]) == 1: continue
			cur_minus = list(cur_pops[old_pop])
			cur_minus.remove(ind)
			if tuple(cur_minus) not in calculated:
				cur_minus_opt = optimize_params_one_pop(freq,reads,cur_minus,detail=detail)
				calculated[tuple(cur_minus)] = cur_minus_opt
			else:
				cur_minus_opt = calculated[tuple(cur_minus)]
			for j in range(len(cur_pops)):
				if ind in cur_pops[j]:
					continue
				cur_test = sorted([item for sublist in [cur_pops[j],[ind]] for item in sublist])
				if tuple(cur_test) not in calculated:
					cur_test_opt = optimize_params_one_pop(freq,reads,cur_test,detail=detail)
					calculated[tuple(cur_test)] = cur_test_opt
				else:
					cur_test_opt = calculated[tuple(cur_test)]
				new_lnL = cur_test_opt[1] + cur_minus_opt[1] #the one it's in now, the old one without it
				old_lnL = cur_opts[j][1] + cur_opts[old_pop][1]
				lamb = 2*(old_lnL-new_lnL)
				print [cur_test, cur_minus], [cur_pops[j], cur_pops[old_pop]], lamb
				if lamb > best_lamb:
					best_lamb = lamb
					best_pop = cur_test
					best_test_opt = cur_test_opt
					best_j = j
			if best_lamb > 0:
				cur_opts[best_j] = best_test_opt
				cur_opts[old_pop] = cur_minus_opt
				cur_pops[best_j] = best_pop
				cur_pops[old_pop] = cur_minus
			print cur_pops, sum(map(lambda x: x[1], cur_opts))
		print cur_pops, sum(map(lambda x: x[1], cur_opts))
	return cur_pops, cur_opts
	

def cluster_join(freq,reads,eps=1e-4,detail=False):
	num_ind = len(reads)
	cur_pops = []
	for i in range(num_ind):
		cur_pops.append([i])
	cur_opts = optimize_pop_params(freq,reads,cur_pops,detail=detail-1)
	any_to_merge = True
	calculated = {}
	while any_to_merge:
		best_merge = []
		best_lambda = 0
		best_improv = 0
		any_to_merge = False
		for i in range(len(cur_pops)-1):
			for j in range(i+1,len(cur_pops)):
				cur_test = [item for sublist in [cur_pops[i],cur_pops[j]] for item in sublist]
				print cur_test
				if tuple(cur_test) not in calculated:
					cur_test_opt = optimize_params_one_pop(freq,reads,cur_test,detail=detail-1)
					calculated[tuple(cur_test)] = cur_test_opt
				else:
					cur_test_opt = calculated[tuple(cur_test)]
				old_lnL = cur_opts[i][1] + cur_opts[j][1]
				new_lnL = cur_test_opt[1]
				new_lambda = 2*(old_lnL-new_lnL)
				rel_improv = -(new_lnL/old_lnL-1)
				if detail: print old_lnL, new_lnL, new_lambda, rel_improv
				#if new_lambda > best_lambda:
				if rel_improv > eps and rel_improv > best_improv:
					best_merge = cur_test
					best_i = i
					best_j = j
					best_opt = cur_test_opt
					best_lambda = new_lambda
					best_improv = rel_improv
					any_to_merge = True
		if best_lambda == 0: break
		print best_merge, best_lambda, best_improv
		cur_opts[best_i] = best_opt
		cur_opts.pop(best_j)	
		cur_pops[best_i] = best_merge
		cur_pops.pop(best_j)
		print cur_pops
	return cur_pops, cur_opts

def optimize_pop_params(freq,reads,pops,detail=False):
	min_a, min_d, read_like = bound_and_precompute_read_like(reads)
	freqs, read_lists = make_read_dict_by_pop(freq,reads,pops)
	opts = []
	for i in range(len(pops)):
		if np.array_equal(pops[i],[]):
			opts.append(None)
			continue
		print "Processing pop %d: %s"%(i,str(pops[i]))
		cur_opt = opt.fmin_l_bfgs_b(func=lambda x: -sum(compute_GT_like_DP(read_lists[i],freqs,x[0],x[1],read_like,min_a,min_d,detail=detail)), x0 = st.uniform.rvs(size=2), approx_grad=True,bounds=[[.00001,10],[.00001,10]])#,epsilon=.001, factr=10, pgtol=1e-10) 
		opts.append(cur_opt)
	return opts

def optimize_params_one_pop(freq,reads,pop,detail=False):
	all_inds = set(range(len(reads)))
	not_in_pop = all_inds.difference(pop)
	freqs, read_lists = make_read_dict_by_pop(freq,reads,[pop,list(not_in_pop)])
	min_a, min_d, read_like = bound_and_precompute_read_like(reads)
	cur_opt = opt.fmin_l_bfgs_b(func=lambda x: -sum(compute_GT_like_DP(read_lists[0],freqs,x[0],x[1],read_like,min_a,min_d,detail=detail)), x0 = st.uniform.rvs(size=2), approx_grad=True,bounds=[[.00001,10],[.00001,10]],epsilon=.001)#, factr=10, pgtol=1e-10) 
	return cur_opt
	

def compute_GT_like_DP(reads,freq,t1,t2,precompute_read_prob,min_a,min_d,detail=False):
	if reads[0][0].ndim == 1:
		n_diploid = 1
	else:
		n_diploid = len(reads[0][0])
	n_haploid = 2*n_diploid
	sampling_prob = compute_Ehet(freq,n_haploid,t1,t2)
	like_per_freq = []
	for i in range(len(freq)):
		read_prob_per_site = compute_all_read_like(reads[i],precompute_read_prob,min_a,min_d)
		read_prob = read_prob_DP(read_prob_per_site)
		like_per_freq.append(sum(np.log(np.dot(read_prob,sampling_prob[i]))))
	if detail: print t1, t2, -sum(like_per_freq)
	return like_per_freq

