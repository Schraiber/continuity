from ancient_genotypes import *
from numpy import *
import pandas
import cPickle
from joblib import Parallel, delayed

def sim_and_infer(num_ind, i, coverage=1):
	freq_sim, GT_sim, reads_sim =  ancient_sample_many_pops(num_modern=1000,anc_pop = [1], anc_per_pop = [num_ind],  anc_time=[300],split_time=[400],Ne0=10000,NeAnc=[1000],mu=1.25e-8,length=500,num_rep=100000,coverage=coverage,error=st.expon.rvs(size=num_ind,scale=.05,random_state=i),seed=i)
	pop = [range(num_ind)]
	params_pop_sim = optimize_pop_params_error(freq_sim,reads_sim,pop,detail=False)
	return params_pop_sim

cov = [.5,1,2,4,8]
results = []
for cur_cov in cov:
	print cur_cov
	results.append([])
	for num_ind in range(1,11):
		print num_ind
		results[-1].append(Parallel(n_jobs=50)(delayed(sim_and_infer)(num_ind,i, coverage=cur_cov) for i in random.randint(1000000,size=200)))
cPickle.dump(results,open("results_coverage_10_ind_200_rep.pickle","w"))
