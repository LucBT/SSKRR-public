import numpy as np
import torch
# import tensorflow as tf
# import tensorflow_probability as tfp
# tfd = tfp.distributions
# psd_kernels = tfp.math.psd_kernels

def compute_RMS_from_data(_x):
    """
    Compute the root mean square distance from a set of positions.
    """
    squared_distance = []
    try: # d-dimension
        n_obs, dimension = np.shape(_x)
        for i in range(n_obs):
            for j in range(i+1, n_obs):
                dist = 0
                for dim in range(dimension):
                    dist = dist + (_x[i, dim] - _x[j, dim])**2
                squared_distance.append(dist)

        RMS = np.sqrt( (2 / (n_obs * (n_obs-1)))*np.sum(squared_distance) )
        print("RMS computed from {} observations of dimension {}: {}".format(n_obs, dimension, RMS))
        return RMS
    except: # 1-dimension
        n_obs = np.size(_x)
        dimension = 1
        for i in range(n_obs):
            for j in range(i+1, n_obs):
                dist = 0
                dist = dist + (_x[i] - _x[j])**2
                squared_distance.append(dist)
        RMS = np.sqrt( (2 / (n_obs * (n_obs-1)))*np.sum(squared_distance) )
        print("RMS computed from {} observations of dimension {}: {}".format(n_obs, dimension, RMS))
        return RMS
    return
def scaling_inputs(_x, _length_scale):
    """
    Scale the inputs by _length_scale
    """
    _x_scaled = np.zeros((np.shape(_x)))
    dimension = np.shape(_x)[1]
    if isinstance(_x, list):
        _x_scaled[k] = _x[k] / _length_scale[0]
    else:
        for dim in range(dimension):
            _x_scaled[:, dim] = _x[:, dim] / _length_scale[dim]

    return _x_scaled

def standardize(_x, _mean, _std):
    """
    Return the standardized version of a vector _x_std with respect to the mean _mean and the std _std, i.e. with zero mean and unit std.
    _mean : sample mean
    _std : sample standard deviation
    """
    try: # d-dimension
        n_obs, dimension = np.shape(_x)
        _x_std = np.zeros((np.shape(_x)))
        for dim in range(dimension):
            _x_std[:, dim] = (_x[:, dim] - _mean[dim]) / (_std[dim])
        return _x_std
    except : # 1-d dimension
        _x_std = np.zeros((np.shape(_x)))
        dimension = 1
        for dim in range(dimension):
            _x_std = (_x - _mean) / (_std)
        return _x_std

def unstandardize(_x_std, _mean, _std):
    """
    Return the standardized version of a vector _x_std with respect to the mean _mean and the std _std, i.e. with zero mean and unit std.
    _mean : sample mean
    _std : sample standard deviation
    """
    try: # d-dimension
        n_obs, dimension = np.shape(_x_std)
        _x_unstd = np.zeros((np.shape(_x_std)))
        for dim in range(dimension):
            _x_unstd[:, dim] = (_x_std[:, dim])*(_std[dim]) + _mean[dim]
        return _x_unstd

    except: # 1-d dimension
        _x_unstd = np.zeros((np.shape(_x_std)))
        dimension = 1
        for dim in range(dimension):
            _x_unstd = (_x_std)*_std + _mean
        return _x_unstd


def normalize(_x, minx, maxx, a , b, printing = True):
    n_obs, dimension = np.shape(_x)
    _x_norm = np.zeros(np.shape(_x))
    mini, maxi = minx, maxx

    check_a = isinstance(a, float) or isinstance(a, int)
    check_b = isinstance(b, float) or isinstance(b, int)
    if (check_a and check_b):
        a, b = [a]*dimension, [b]*dimension
    else:
        pass
    
    if printing == True:
        print(f"From {minx} / {maxx} to {a} / {b}")
    for i in range(n_obs):
        for j in range(dimension):
            _x_norm[i, j] = (b[j] - a[j])*((_x[i, j] - mini[j])/(maxi[j] - mini[j])) + a[j]
    return _x_norm

def predict_GPRegression(_index_points, _observation_index_points, _observations, _kernel, _jitter):
    """
    Compute the GP regressor using TensorFlow. The inputs must be standardized before using it.
    To get the mean of the GP process after conditioning, take .mean()
    To get the std of the GP process after conditioning, take .std()
    This function is quite long if one is looking for predicting at one point at a time.
    """
    if isinstance(_index_points, (list, tuple)):
        dimension = len(_index_points)
        _index_points = np.array(_index_points).reshape((-1, dimension))
    _N_obs = np.shape(_observations)[0]

    _GPR = tfd.GaussianProcessRegressionModel(
			kernel= _kernel,
			index_points= _index_points,
			observation_index_points= _observation_index_points,
			observations=np.reshape(_observations, (_N_obs, )),
			jitter=_jitter)

    return _GPR

def predict_fast_kriging(_index_points, _observation_index_points, _kernel, _omega):
    """
    Use this function if you want to predict at an unobserved point when using parametric KF or normal GPR.
    F*(x) = K(x,X)*_omega
    where _omega = (K(X, X) + λI_N)^(-1) Y_N
    and K(x,X) = kernel_obs
    This function is much faster than TensorFlow if one is looking for predicting at one point at a time.
    """
    if isinstance(_index_points, (list, tuple)):
        dimension = len(_index_points)
        _index_points = np.array(_index_points).reshape((-1, dimension))
    kernel_obs = _kernel.apply(_index_points, _observation_index_points)
    prediction = np.matmul(kernel_obs, _omega)
    return np.float64(prediction)

def predict_fast_kriging_fn(_index_points, _observation_index_points, _kernel, _omega, _omega_fn, F_N):
    """
    Use this function if you want to predict at an unobserved point when using non-parametric KF.
    F*(x) = K(x,X)*_omega
    where _omega = (K(X, X) + λI_N)^(-1) Y_N
    and K(x,X) = kernel_obs
    This function is much faster than TensorFlow if one is looking for predicting at one point at a time.
    """
    if isinstance(_index_points, (list, tuple)):
        dimension = len(_index_points)
        _index_points = np.array(_index_points).reshape((-1, dimension))

    f_n = predict_fast_kriging(_index_points, _observation_index_points, _kernel, _omega_fn)
    kernel_obs = _kernel.apply(f_n, F_N)
    prediction = np.matmul(kernel_obs, _omega)
    return np.float64(prediction)

def create_pif_pic(num_obs, Nf, Nc, seed = None):
    import random
    if seed != None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Pif points
    Pitot = np.full((num_obs), False)
    Pitot[:Nf] = True
    rng.shuffle(Pitot)
    Pif = Pitot

    # Pic points
    Pitot = np.full((num_obs), False)
    Pitot[rng.choice(np.where(Pif == True)[0], size = Nc, replace = False)] = True
    Pic = Pitot
    # Pif = sorted([i for i in rng.sample(range(num_obs), Nf)])
    # Pic = sorted(rng.sample(Pif, Nc))
    return Pif, Pic

def compute_rho_tf(_Xf, _Xc, Ff, Fc, _kernel, _nugget):
	#* Xf points
	Nf = np.size(Ff)
	Kf = _kernel.matrix(_Xf, _Xf) + _nugget*np.identity(Nf)
	#* Xc points
	Nc = np.size(Fc)
	Kc = _kernel.matrix(_Xc, _Xc) + _nugget*np.identity(Nc)
	#* Compute rho
	rho = 1 - (tf.transpose(Fc) @ tf.linalg.inv(Kc) @ Fc) / (tf.transpose(Ff) @ tf.linalg.inv(Kf) @ Ff)
	if rho < 0 or rho > 1:
		raise ValueError("Error in rho: {}".format(rho[0][0]))
	return rho[0][0]

def compute_rho_tf_gamma(_Xf, _Xc, Ff, Fc, _gamma, _nugget, kernel_name = "Gaussian"):
	_length_scale = _gamma / np.sqrt(2)
	if kernel_name == "Gaussian":
		_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(1., _length_scale)
	else:
		print("No other Kernel choice available")
		raise ValueError
	#* Xf points
	Nf = np.size(Ff)
	Kf = _kernel.matrix(_Xf, _Xf) + _nugget*np.identity(Nf)
	#* Xc points
	Nc = np.size(Fc)
	Kc = _kernel.matrix(_Xc, _Xc) + _nugget*np.identity(Nc)

	#* Compute rho
	rho = 1 - (tf.transpose(Fc) @ tf.linalg.inv(Kc) @ Fc) / (tf.transpose(Ff) @ tf.linalg.inv(Kf) @ Ff)
	if rho < 0 or rho > 1:
		raise ValueError("Error in rho: {}".format(rho[0]))
	return rho[0][0]

def compute_rho_tf_sigma(_Xf, _Xc, Ff, Fc, _sigma, _gamma, _nugget, kernel_name = "Gaussian"):
	_length_scale = _gamma / np.sqrt(2)
	if kernel_name == "Gaussian":
		_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(_sigma, _length_scale)
	else:
		print("No other Kernel choice available")
		raise ValueError
	#* Xf points
	Nf = np.size(Ff)
	Kf = _kernel.matrix(_Xf, _Xf) + _nugget*np.identity(Nf)
	#* Xc points
	Nc = np.size(Fc)
	Kc = _kernel.matrix(_Xc, _Xc) + _nugget*np.identity(Nc)

	#* Compute rho
	rho = 1 - (tf.transpose(Fc) @ tf.linalg.inv(Kc) @ Fc) / (tf.transpose(Ff) @ tf.linalg.inv(Kf) @ Ff)
	if rho < 0 or rho > 1:
		raise ValueError("Error in rho: {}".format(rho[0]))
	return rho[0][0]

def compute_LOO_error(F_obs, Kernel_obs, nugget):
    """
    Compute the Leave-one-out error of Kriging
    """
    N_OBS = np.shape(F_obs)[0]
    K = Kernel_obs + nugget*np.eye(N_OBS)
    Kinv = np.linalg.inv(K)
    diag_Kinv_2 = np.diag(np.diag(Kinv)**(-2))
    error = np.transpose(F_obs) @ Kinv @ diag_Kinv_2 @ Kinv @ F_obs
    return np.float64(error / N_OBS)

def compute_LOO_error_torch(F_obs, Kernel_obs, nugget):
    """
    Compute the Leave-one-out error of Kriging
    """
    N_OBS = F_obs.shape[0]
    K = Kernel_obs + nugget*torch.eye(N_OBS)
    Kinv = torch.linalg.inv(K)
    diag_Kinv_2 = torch.diag(torch.diag(Kinv)**(-2))
    error = F_obs.T @ Kinv @ diag_Kinv_2 @ Kinv @ F_obs
    return error / N_OBS

def compute_test_error_RMSE_torch(X_test, F_test, _gpr_torch):
    error_RMSE = 0
    mean, std = _gpr_torch(X_test, full_cov=False, noiseless=True)
    temp = (mean - F_test)**2
    temp2 = torch.mean(temp)
    error_RMSE = torch.sqrt(temp2)
    return error_RMSE