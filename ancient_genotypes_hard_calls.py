
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


def expected_het_anc(x0,t):
	return 1.0/(3.0/2.0+(2*x0-1)/(1+np.exp(2*t)-2*x0))

def expected_het_split(x0,t1,t2):
	return 1.0/(1.0/2.0+(np.exp(2*t1+t2))/(1+np.exp(2*t1)-2*x0))

def expected_moments_split(x0,t1,t2):
	Ehet = np.exp(-3.*t1-t2)*(1.+np.exp(2.*t1)-2.*x0)*x0
	Eder = 1./2.*np.exp(-3.*t1-t2)*(2.*x0+np.exp(2.*t1)*(2.*np.exp(t2)-1.)-1.)*x0
	Eanc = 1.0  - Ehet - Eder
	return Eanc, Ehet, Eder

def get_numbers_from_dict(anc_dict):
	het = []
	hom = []
	for freq in sorted(anc_dict.keys()):
		het.append(anc_dict[freq][1])
		hom.append(anc_dict[freq][2])
	return np.array(het), np.array(hom)

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
		plt.plot(freqs,hetAnc,'r',label="anc, t = %0.2f"%tAnc)
		plt.plot(freqs,hetSplit,'y',label="split, t1 = %0.2f t2 = %0.2f"%(t1,t2))
		plt.xlabel("Frequency")
		plt.ylabel("Proportion of het sites")
		plt.legend()
		plt.title(title)
	return ancTest,splitTest
