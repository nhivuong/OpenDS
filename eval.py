# Source: https://github.com/tarunsharma1/saliency_metrics/blob/master/salience_metrics.py
def discretize_gt(gt):
	import warnings
	warnings.warn('can improve the way GT is discretized')
	return gt/255

def normalize_map(s_map):
	# normalize the salience map (as done in MIT code)
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map

def auc_judd(s_map,gt):
	# ground truth is discrete, s_map is continous and normalized
	gt = discretize_gt(gt)
	# thresholds are calculated from the salience map, only at places where fixations are present
	thresholds = []
	for i in range(0,gt.shape[0]):
		for k in range(0,gt.shape[1]):
			if gt[i][k]>0:
				thresholds.append(s_map[i][k])

	
	num_fixations = np.sum(gt)
	# num fixations is no. of salience map values at gt >0


	thresholds = sorted(set(thresholds))
	
	#fp_list = []
	#tp_list = []
	area = []
	area.append((0.0,0.0))
	for thresh in thresholds:
		# in the salience map, keep only those pixels with values above threshold
		temp = np.zeros(s_map.shape)
		temp[s_map>=thresh] = 1.0
		assert np.max(gt)==1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
		assert np.max(s_map)==1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
		num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
		tp = num_overlap/(num_fixations*1.0)
		
		# total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
		# this becomes nan when gt is full of fixations..this won't happen
		fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
		
		area.append((round(tp,4),round(fp,4)))
		#tp_list.append(tp)
		#fp_list.append(fp)

	#tp_list.reverse()
	#fp_list.reverse()
	area.append((1.0,1.0))
	#tp_list.append(1.0)
	#fp_list.append(1.0)
	#print tp_list
	area.sort(key = lambda x:x[0])
	tp_list =  [x[0] for x in area]
	fp_list =  [x[1] for x in area]
	return np.trapz(np.array(tp_list),np.array(fp_list))