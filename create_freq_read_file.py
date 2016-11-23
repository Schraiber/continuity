import pysam
import argparse
import glob
import sys

parser = argparse.ArgumentParser("Get counts of reads from ancient low coverage data relative to frequencies from a modern reference population")
parser.add_argument("-v", required = True, help = "VCF of modern reference individuals")
parser.add_argument("-r", required = True, help = "List of individuals in modern reference population")
parser.add_argument("-c", type=float, default = 1.0,  help = "Minimum number of individuals in reference population to include site")
parser.add_argument("-e", default="", help = "Comma-separated list of excluded chromosomes")
parser.add_argument("-b", required = True, help = "Path to BAM files")
parser.add_argument("-s", required = True, help = "Path to SNP file")

args = parser.parse_args()

#open bams
#TODO: Figure out why it doesn't seem to be doing every dude
bams = glob.glob("%s/*.bam"%args.b)
bam_names = []
bam_files = []
for bam in bams:
	bam_names.append(bam.split("/")[-1].split(".")[0])
	bam_files.append(pysam.AlignmentFile(bam))

sys.stderr.write("Processing %i samples\n"%len(bam_names))

#write the header to the output
header_list = ["Chrom\tPos\tAF"]
for name in bam_names:
	header_list.append("%s_der\t%s_anc\t%s_other"%(name,name,name))
header = '\t'.join(header_list)
sys.stdout.write("%s\n"%header)

#open VCF 
#TODO: Need to open a whole path of VCFs
vcf =  pysam.VariantFile(args.v)

#open SNP file
SNPs = open(args.s)

#open inds in reference pop
inds = []
for line in open(args.r):
	inds.append(line.strip())

#loop over SNPs
for SNP in SNPs:
	SNP_split = SNP.split()
	chrom = SNP_split[1]
	pos = int(SNP_split[3])
	for variant in vcf.fetch(str(chrom), pos-1, pos):
		if variant.pos != pos: break #for now, skip ones where any with the wrong pos are returned...
		ref = variant.ref
		alt = variant.alts
		if len(alt) > 1: break
		alt = alt[0]
		try:
			AAinfo = variant.info['AA']
		except KeyError:
			break
		AA = AAinfo.split("|")[0]
		if AA not in [ref, alt]: break
		count = 0.
		good_alleles = 0.
		for ind in inds:
			alleles = variant.samples[ind].allele_indices
                	filtered_alleles = [a for a in alleles if a is not None]
                	count += sum(filtered_alleles)
                	good_alleles += len(filtered_alleles)
		freq = count/good_alleles
		if freq == 0 or freq == 1: break
		if AA == alt:
			der = ref 
			freq = 1-freq
		else:
			der = alt
		sys.stdout.write("%s\t%d\t%f"%(chrom,pos,freq))
		for i,bam in enumerate(bam_files):
			sys.stdout.write("\t")
			der_count, anc_count, other_count = 0, 0, 0
			try:
				pileup = bam.pileup(str(chrom), pos-1, pos)
			except ValueError:
				sys.stdout.write("%d\t%d\t%d"%(der_count,anc_count,other_count))
				break	
			for pileupcolumn in pileup: 
				if pileupcolumn.pos != pos: continue
				for pileupread in pileupcolumn.pileups:
					if pileupread.is_del or pileupread.is_refskip: continue
					base = pileupread.alignment.query_sequence[pileupread.query_position]
					#TODO: Figure out why I'm getting so many other counts
					if base == der: der_count += 1
					elif base == AA: anc_count += 1
					else: other_count += 1
			sys.stdout.write("%d\t%d\t%d"%(der_count,anc_count,other_count))
		sys.stdout.write("\n")


