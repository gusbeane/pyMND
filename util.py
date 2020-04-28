import numpy as np 

def rejection_sample(fn, maxval, N, xrng=[0, 1], overshoot=2., dtype=np.longdouble, fn_args={}):
	xwidth = xrng[1] - xrng[0]
	xstart = xrng[0]

	sample_list = np.array([])

	N_needed = np.copy(N)
	while(N_needed > 0):
		print('drawing '+str(int(overshoot*N_needed)) + ' samples...')
		x = np.random.rand(int(overshoot*N_needed))
		y = np.random.rand(int(overshoot*N_needed))
	
		x = np.multiply(x, xwidth)
		x = np.add(x, xstart)
		y = np.multiply(y, maxval)

		fn_eval = fn(x, **fn_args)

		keys = np.where(fn_eval > maxval)[0]
		assert len(keys)==0, "Maxval provided is not big enough, maxval="+str(maxval)+" but got max(fn)="+str(np.max(fn_eval))
		
		keys = np.where(y < fn_eval)[0]
		N_this_iter = len(keys)
		if N_this_iter > 0:
			sample_list = np.concatenate((sample_list, x[keys]))
	
		N_needed -= N_this_iter

	return sample_list[:N]


