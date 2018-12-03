import signal

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)


from ancient_genotypes_simulation import *
from ancient_genotypes import *
from numpy import *
import pandas
import cPickle
from joblib import Parallel, delayed

time_out_errors = open("admixture_cont_time_out.txt","w")

def sim_ghost_admixture(num_ind, i, f, coverage=1):
	print [coverage, num_ind, f]
	freq_sim, GT_sim, reads_sim = ancient_sample_ghost_mix(anc_pop = 0, anc_num = num_ind, Ne1=1000,f=f, anc_time = 300, mix_time = 200, length=500,num_rep=100000,coverage=coverage,error=st.expon.rvs(size=num_ind,scale=.05,random_state=i),seed = i)
	freqs_sim, read_list_sim = get_read_dict(freq_sim,reads_sim)
	signal.alarm(21600) 
	try:
		params_pop_sim_free = optimize_pop_params_error_parallel(freqs_sim,read_list_sim,num_core=1,detail=0,continuity=False)
		params_pop_sim_continuity = optimize_pop_params_error_parallel(freqs_sim,read_list_sim,num_core=1,detail=0,continuity=True)
	except TimeoutException:
		time_out_errors.write("Failed, num_ind: %i, i: %i, f: %f, coverage: %f\n"%(num_ind,i,f,coverage)) 
		params_pop_sim_free = None
		params_pop_sim_continuity = None
	return [params_pop_sim_free,params_pop_sim_continuity]

cov = [.5,4]
results = []
for cur_cov in cov:
	print cur_cov
	results.append([])
	for num_ind in [1,5]:
		results[-1].append([])
		for f in np.linspace(0.,.5,num=10):
			results[-1][-1].append(Parallel(n_jobs=50)(delayed(sim_ghost_admixture)(num_ind,i,f=f,coverage=cur_cov) for i in random.randint(1000000,size=200)))
			cPickle.dump(results,open("results_ghost_admixture_cont_half_4_coverage_1_5_ind_f_0_half_rep.pickle","w"))
