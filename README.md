# Inferring the relationship of ancient and modern populations

## Installation

For now, just download/clone the repo. There's a lot of junk in here, and I'll clean it up eventually. The key file is `ancient_genotypes.py`. 

There are several dependencies. You will need `numpy`, `matplotlib`, `scipy`, `joblib`, and `pandas`. 

## Data format

The current data format is a tab separated file in which each line is a SNP, the first column is the chromosome name, second column position and the third column is *derived* allele frequency in reference modern population (thus, you must be able to polarize your alleles). NB: DO NOT include alleles that are frequency 0 or frequency 1 in the reference modern popuation. Subsequent columns are three per ancient sample, with the counts of the number of reads at that site with the derived allele, the ancestral allele, and some other allele, respectively. There is a required header: 

```
Chrom	Pos	AF	anc1_der	anc1_anc	anc1_other	anc2_der	anc2_anc	anc2_other	...
```

where `anc1`, `anc2`, ... are sample identifiers for the ancient samples. For example, the following excerpt shows how the file should look for 5 SNPs with 2 ancient individuals

```
Chrom	Pos	AF	I0412_der	I0412_anc	I0412_other	I1277_der	I1277_anc	I1277_other
1	891021	0.943925	43	0	0	5	0	0
1	903426	0.280374	0	30	0	0	1	0
1	949654	0.911215	0	16	0	0	0	0
1	1005806	0.196262	1	23	0	0	3	0
1	1018704	0.481308	155	0	2	2	3	0
1	1021415	0.275701	5	0	0	1	1	0
1	1021695	0.285047	23	0	0	0	1	0
1	1031540	0.696262	0	0	0	0	0	0
1	1045331	0.023364	0	31	0	0	0	0
```

You can also indicate the reference population sample size and number of derived alleles. Intead of an AF column, you can add two new clumns, ``kref`` and ``nref``, which are the counts of derived alleles and the number of non-missing chromosomes in the references population. So, for example the above file might instead look like

```
Chrom	Pos	kref	nref	I0412_der	I0412_anc	I0412_other	I1277_der	I1277_anc	I1277_other
1	891021	94	100	43	0	0	5	0	0
1	903426	28	100	0	30	0	0	1	0
1	949654	91	100	0	16	0	0	0	0
1	1005806	19	100	1	23	0	0	3	0
1	1018704	48	100	155	0	2	2	3	0
1	1021415	27	100	5	0	0	1	1	0
1	1021695	28	100	23	0	0	0	1	0
1	1031540	69	100	0	0	0	0	0	0
1	1045331	2	100	0	31	0	0	0	0
```
*NOTE THAT THE COLUMNS ``kref`` AND ``nref`` NEED TO HAVE EXACTLY THOSE HEADERS*.

## Specifying ancient panels

You also need to specify the panels that the ancient individuals belong to (i.e. a prior populations) using an eigenstrat-format ind file, with each line corresponding to a sample and 3 columns, sample name, sex (not used, so it doesn't actually have to be correct, but the column is required), and population name. The individual names need to match the individual names in the input data. Yes, you even need to do this if you only have one individual. For instance, for the above data, we have

```
I0412 M  Iberia_EN
I1277 M Iberia_Chalcolithic
```

## Reading data into Python

You can use the function ``parse_reads_by_pop()`` to read in your data. It has 3 input arguments

1. ``read_file_name``: the path to the file with the counts of reads from the bam file 
2. ``ind_file_name``: the path to the ind file with the ancient samples in it
3. ``cut off``: the minimum fraction of ancient samples that should have at least one read at a site to count that site (default 0)

This fuction returns *six* different objects.

1. ``unique_pops``: all the unique ancient panels from your ind file
2. ``inds``: the individuals in the reads file
3. ``label``: the population to which each individual belongs (indices correspond to ``unique_pops``)
4. ``pops``: a list of lists, specifying population membership 
5. ``freqs``: a list of the unique allele frequencies in the reference population
6. ``read_lists``: a list of lists, with the read data for all ancient pops

For example, you can do this

```
from ancient_genotypes import *
unique_pops, inds, label, pops, freqs, read_lists = parse_reads_by_pop("path/to/reads/file.reads","/path/to/ind/file.ind",cutoff=0)
```

## Filtering for coverage in the ancient sample

For various reasons, such as cryptic structural variation, you may have some sites with unusually high/low coverage in your ancient samples. These sites can severely mislead analyses.  A common approach is to remove sites that fall in the tails of the coverage distribution. You can do that once you've read in your data by using the function ``coverage_filter()``. It will set any sites that violate a coverage filter in an individual to have 0 coverage. It takes 3 arguments:

1. ``read_lists``: The ``read_lists`` object that comes out of ``parse_reads_by_pop()``
2. ``min_cutoff``: sites with coverage less than this percentile will be removed (default 2.5)
3. ``max_cutoff``: sites with coverage more than this percentile will be removed (default 97.5)

This function operates on ``read_lists`` *IN PLACE*, meaning that ``read_lists`` will be modified. The function returns a single value, which is a list of the coverage cutoffs for each individual in each population.

## Fitting a prior distribuiton to the discrete reference allele frequencies

If you use discrete reference allele frequencies (i.e. you specify the ``kref`` and ``nref`` columns), it is strongly recommended that you fit a beta prior to the allele frequencies. This essentially smooths out the discrete frequencies, can be very helpful for small reference sample sizes. To do so, use the function ``get_beta_params()``. This funciton has 3 input arguments:

1. ``freqs``: The ``freqs`` object returned by ``parse_reads_by_pop()``
2. ``read_lists``: the ``read_lists`` object returned by ``parse_reads_by_pop()``
3. ``min_samples``: the minimum number of samples to be used when fitting beta parameters. This is useful if you have varying missingness across reference sites. The default setting is 15, meaning that only sites where ``nref`` > 15 will be used. It's not recommended to have ``min_samples`` < 10.

This returns two numbers, the maximum likelihood estimates of the ``alpha`` and ``beta`` priors on the allele frequency distribution. For example,

```
alpha, beta = get_beta_params(freqs, read_lists, min_samples = 15)
```

## Estimating parameters

There is a function ``optimize_pop_params_error_parallel()`` that will fit the model to your data for every population. It has five arguments regardless of whether you have continuous or discrete reference allele frequencies.:

1. ``freqs``: the ``freqs`` object that's output from ``parse_reads_by_pop()``
2. ``read_lists``: the ``read_lists`` object that's output from ``parse_reads_by_pop()``
3. ``num_core``: the number of cores to use. Each different population is farmed out to a different core.
4. ``detail``: whether to print some updates as the optimization is going (default ``False``)
5. ``continuity``: whether to optimize the parameters while holding `t2 = 0` (i.e. finding the best fitting parameters assuming population continuity with the ancient sample) (default False)

If you have discrete reference allele frequencies, you also need to specify the alpha and beta parameters:

6. ``alpha``: the alpha parameter of the beta prior on allele frequencies in the reference population
7. ``beta``: the beta parameter of the beta prior on allele frequencies in the reference population

This will return a list of scipy.optimize objects, each one corresponding to a population in ``pops``. The important parts of each object are the 0th entry, which are the parameters of the model, and the 1st entry, which is the *negative* log likelihood of the model. The parameters are in the order ``t1``, ``t2``, ``error_for_ind_1``, ``error_for_ind_2``, and so on. 

## Testing continuity

If you want to do a likelihood ratio test for continuity you should run ``optimize_pop_params_error_parallel()`` *twice*, once with ``continuity = False`` and the second time with ``continuity = True``. Then, you compute the likelihood ratio statstic as ``2*(likelihood_cont_true - likelihood_cont_false)`` and that is distributed as chi-squared with 1 degree of freedom under the null. 

For example, continuing with the variables above

```
import scipy.stats
opts_cont_false = optimize_pop_params_error_parallel(freqs,read_lists,pops,continuity=False)
opts_cont_true = optimize_pop_params_error_parallel(freqs,read_lists,pops,continuity=True)
likelihood_false = np.array([-x[1] for x in opts_cont_false]) #minus sign is because scipy.optimize minimizes the negative log likelihood
likelihood_true = np.array([-x[1] for x in opts_cont_true])
LRT = 2*(likelihood_false - likelihood_true)
p_vals = scipy.stats.chi2.logsf(LRT,1) #returns the LOG p-values
```

 
