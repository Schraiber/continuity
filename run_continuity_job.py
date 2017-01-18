from ancient_genotypes import *
from numpy import *
import pandas
import cPickle
from joblib import Parallel, delayed

def sim_and_infer_continuity(num_ind, i, time, coverage=1):
	print [coverage, num_ind, time]
	freq_sim, GT_sim, reads_sim =  ancient_sample_many_pops(num_modern=1000,anc_pop = [1], anc_per_pop = [num_ind],  anc_time=[time],split_time=[400],Ne0=10000,NeAnc=[1000],mu=1.25e-8,length=500,num_rep=100000,coverage=coverage,error=st.expon.rvs(size=num_ind,scale=.05,random_state=i),seed=i)
	pop = [range(num_ind)]
	params_pop_sim_free = optimize_pop_params_error(freq_sim,reads_sim,pop,detail=False)
	params_pop_sim_continuity = optimize_pop_params_error(freq_sim,reads_sim,pop,continuity = True, detail=False)
	return [params_pop_sim_free,params_pop_sim_continuity]

cov = [.5,4]
results = []
for cur_cov in cov:
	print cur_cov
	results.append([])
	for num_ind in [1,5]:
		results[-1].append([])
		for time in np.linspace(0.,399.,num=10):
			results[-1][-1].append(Parallel(n_jobs=50)(delayed(sim_and_infer_continuity)(num_ind,i,time=time,coverage=cur_cov) for i in random.randint(1000000,size=200)))
			cPickle.dump(results,open("results_continuity_half_4_coverage_1_5_ind_200_rep.pickle","w"))
