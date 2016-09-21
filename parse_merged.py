import argparse
import pysam
import pickle
import sys

parser = argparse.ArgumentParser("Get counts of genotypes in test individuals correspondning to frequencies in a reference population. Expects a merged VCF with all samples")
parser.add_argument("-v", required = True, help = "Merged VCF")
parser.add_argument("-t", required = True, help = "List of test individuals")
parser.add_argument("-r", required = True, help = "List of individuals in reference population")
parser.add_argument("-a", required = True, help = "Individual to use to polarize ancestral allele")
parser.add_argument("-c", type=float, default = 1.0,  help = "Minimum number of individuals in reference population to include site")
parser.add_argument("-e", default="", help = "Comma-separated list of excluded chromosomes")

args = parser.parse_args()

test = []
for line in open(args.t):
	test.append(line.strip())

ref = []
for line in open(args.r):
	ref.append(line.strip())

excluded = args.e.strip().split(",")

vcf = pysam.VariantFile(args.v)

geno_dict = {ind:{} for ind in test}

num = 0

for variant in vcf.fetch():
	if num%10000==0: sys.stderr.write("%s %i\n"%(variant.chrom,variant.pos))
	num += 1
	if variant.chrom in excluded: continue
	anc = variant.samples[args.a].allele_indices
	if None in anc: continue
	if anc[0] != anc[1]: continue
	anc = anc[0]
	count = 0.0
	good_alleles = 0.0
	for ind in ref:
		alleles = variant.samples[ind].allele_indices
		filtered_alleles = [a for a in alleles if a is not None]
		count += sum(filtered_alleles)
		good_alleles += len(filtered_alleles)
	#TODO: Make a minimum cutoff of number of sites
	if good_alleles < args.c: continue
	freq = count/good_alleles
	if freq == 0 or freq == 1: continue
	if anc == 1: freq = 1 - freq
	for ind in test:
		geno = variant.samples[ind].allele_indices
		if None in geno:
			continue
		geno = [a if anc == 0 else 1 - a for a in geno]
		geno = sum(geno)
		if freq not in geno_dict[ind]:
			geno_dict[ind][freq] = [0.0,0.0,0.0]	
		geno_dict[ind][freq][geno] += 1.0
	
pickle.dump(geno_dict, sys.stdout)
