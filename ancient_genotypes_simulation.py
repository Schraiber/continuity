import msprime as msp
import scipy.stats as st
import numpy as np

def ancient_sample_mix_multiple(num_modern=1000,anc_pop = 0, anc_num = 1, anc_time=200,mix_time=300,split_time=400,f=0.0,Ne0=10000,Ne1=10000,mu=1.25e-8,length=1000,num_rep=1000, error = None, coverage=False, seed = None):
	if mix_time > split_time:
		print "mixture occurs more anciently than population split!"
		return None
	if f < 0 or f > 1:
		print "Admixture fraction is not in [0,1]"
		return None
	if error is None:
		error = np.zeros(anc_num)
	samples = [msp.Sample(population=0,time=0)]*num_modern
	samples.extend([msp.Sample(population=anc_pop,time=anc_time)]*(2*anc_num))
	pop_config = [msp.PopulationConfiguration(initial_size=Ne0),msp.PopulationConfiguration(initial_size=Ne1)]
	divergence = [msp.MassMigration(time=mix_time,source=0,destination=1,proportion = f),
			msp.MassMigration(time=split_time,source=1,destination=0,proportion=1.0)]
	sims = msp.simulate(samples=samples,Ne=Ne0,population_configurations=pop_config,demographic_events=divergence,mutation_rate=mu,length=length,num_replicates=num_rep, random_seed = seed)
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
					p_der = cur_GT/2.*(1-error[i])+(1-cur_GT/2.)*error[i]
					derived_reads = st.binom.rvs(num_reads, p_der)
					reads[-1][-1] = (num_reads-derived_reads,derived_reads)
	return np.array(freq), GT, reads

def ancient_sample_ghost_mix(num_modern=1000, anc_pop = 0, anc_num = 1, anc_time = 200, mix_time = 300, split_time_anc = 400, split_time_ghost = 800, f = 0.0, Ne0 = 10000, Ne1 = 10000, NeGhost = 10000, mu = 1.25e-8, length = 1000, num_rep = 1000, error = None, coverage = False, seed = None):
	if mix_time > split_time_anc:
		print "Mixture with ghost occurs more anciently than modern/ancient split"
		return None
	if mix_time > split_time_ghost:
		print "Mixture with ghost occurs more anciently than ghost pop existed!"
		return None
	if f < 0 or f > 1:
		print "Admixture fraction is not in [0, 1]"
		return None
	if error is None:
		error = np.zeros(anc_num)
	samples = [msp.Sample(population = 0, time = 0)]*num_modern
	samples.extend([msp.Sample(population = anc_pop, time = anc_time)]*(2*anc_num))
	pop_config = [msp.PopulationConfiguration(initial_size = Ne0), msp.PopulationConfiguration(initial_size = Ne1), msp.PopulationConfiguration(initial_size = NeGhost)]
	divergence = [msp.MassMigration(time=mix_time, source = 0, destination = 2, proportion = f), msp.MassMigration(time = split_time_anc, source = 1, destination = 0, proportion = 1.0), msp.MassMigration(time = split_time_ghost, source = 2, destination = 0, proportion = 1.0)]
	sims = msp.simulate(samples=samples,Ne=Ne0,population_configurations=pop_config,demographic_events=divergence,mutation_rate=mu,length=length,num_replicates=num_rep, random_seed = seed)
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
					p_der = cur_GT/2.*(1-error[i])+(1-cur_GT/2.)*error[i]
					derived_reads = st.binom.rvs(num_reads, p_der)
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
	sims = msp.simulate(samples=samples,Ne=Ne[0],population_configurations=pop_config,demographic_events=divergence,mutation_rate=mu,length=length,num_replicates=num_rep,random_seed=seed)
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
			#TODO: FIX THIS so the results don't need to be re-parsed
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
					#num_reads = st.geom.rvs(1./coverage)
					p_der = cur_GT/2.*(1-error[ind_num])+(1-cur_GT/2.)*error[ind_num]
					derived_reads = st.binom.rvs(num_reads, p_der)
					reads[ind_num][-1] = (num_reads-derived_reads,derived_reads)

	return np.array(freq), GT, reads	

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
