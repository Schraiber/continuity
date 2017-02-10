from ancient_genotypes import *
from numpy import *
import pandas
import cPickle

freq, reads, inds = parse_reads("../ancient_humans_continuity/Mathieson_SNPs.reads",cutoff=0)
inds_pops = pandas.read_table("/mnt/sda/data/Mathieson_BAMS/MathiesonEtAl_genotypes/full230.ind",delim_whitespace=True,header=None)
unique_pops = unique(inds_pops[ [ind in inds for ind in inds_pops[0]] ][[2]])
label_names = [inds_pops[inds_pops[0] == ind][2].values for ind in inds]
label = [int(where(name == unique_pops)[0]) for name in label_names]
pops = [list(where(array(label)==i)[0]) for i in range(len(unique_pops))]
params_pops = optimize_pop_params_error_parallel(freq,reads,pops,detail=False,num_core=20)
cPickle.dump(params_pops,open("iain_pops.pickle","w"))
