
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
#########################################################################

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

