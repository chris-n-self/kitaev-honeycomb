# 09/05/19
# Chris Self
import copy
import numpy as np
from scipy import linalg as spla

#
# all of these functions take the spectrum as an argument and assume it is ordered from most
# negative to most positions (e.g. [-2,-1,0,1,2]). spectrum can be passed either as a 1d numpy
# array or as a diagonal 2d array.
#

def _safe_T_to_beta(T):
    """ convert T to beta,  """
    if np.isclose(T,0.):
        return 1E16
    else:
        return 1./T

def get_desired_parity(*args,**kwargs):
    """ """

    # unpack kwargs for u arrays
    ux=kwargs['ux']
    uy=kwargs['uy']
    uz=kwargs['uz']

    # allow for open boundaries by discarding zeros in u's'
    nonzero_ux = ux[np.logical_not(np.isclose(ux,0.))]
    nonzero_uy = uy[np.logical_not(np.isclose(uy,0.))]
    nonzero_uz = uz[np.logical_not(np.isclose(uz,0.))]

    desired_parity = np.cumprod(nonzero_ux)[-1]
    desired_parity = desired_parity * (np.cumprod(nonzero_uy)[-1])
    desired_parity = desired_parity * (np.cumprod(nonzero_uz)[-1])

    # use either Q or U to get determinant of Q.
    if ('Q' in kwargs):
        Q = kwargs['Q']
        
        # get determinant of Q
        detQ = spla.det(Q)
        assert np.isclose(np.abs(detQ),1.)

    elif ('U' in kwargs):
        U = kwargs['U']
        
        # get determinant of U
        detU = spla.det(U)
        assert np.isclose(np.abs(detU),1.)
        
        # multiply by phase factor i^(N/2)
        detQ = detU * (1j**ux.size)
        assert np.isclose(np.imag(detQ),0.)
        detQ = np.real(detQ)

    else:
        print("get_desired_parity requires either Q or U as a kwarg.")
        raise ValueError
    
    return desired_parity * detQ

#
# ---------------------------------------------------
# compute log(Z), either projector or unprojected.
# ---------------------------------------------------
#

def get_unprojected_log_Z(spectrum,T):
    """this computes the log of the partition function without the fermionic projection"""
    beta = _safe_T_to_beta(T)

    if len(spectrum.shape)>1:
        spectrum = np.diag(spectrum)

    # get positive half of spectrum, in descending order
    spectrum = np.abs(spectrum[spectrum.size//2:])

    # use logaddexp mode-wise and then sum over modes
    return np.sum(np.logaddexp(-1.*spectrum*beta/2.,-(-1.*spectrum)*beta/2.)),{}

def get_projected_log_Z(spectrum,T,desired_parity):
    """
    this function computes the partition sum for the fermion spectrum at temperature T=1/beta 
    summing over only the many-body states with the correct parity. 

    it RETURNS the logz for the correct parity subspace
    """
    beta = _safe_T_to_beta(T)

    # we need to include log(0) at some points in these computations, numerically we use -1*(some big number)
    # to achieve this
    _big_num = (1E16)
        
    if len(spectrum.shape)>1:
        spectrum = np.diag(spectrum)

    # start from the first mode
    logz_even = beta*spectrum[-1]/2.
    logz_odd = -beta*spectrum[-1]/2.
     
    # iterate through the rest of the modes
    # (accessing the +ve eigenvalues in the second half of the spectrum list)
    number_modes = spectrum.size//2
    for mode_p in range(1,number_modes):
        
        # compute the next level of the iteration
        tmp = logz_even, logz_odd
        logz_even = np.logaddexp( tmp[0] + beta*spectrum[-mode_p-1]/2., tmp[1] - beta*spectrum[-mode_p-1]/2. )
        logz_odd = np.logaddexp( tmp[1] + beta*spectrum[-mode_p-1]/2., tmp[0] - beta*spectrum[-mode_p-1]/2. )

    # return the desired parity
    if np.rint(desired_parity)==1:
        return logz_even,{}
    else:
        return logz_odd,{}

#
# ---------------------------------------------------------------------
# compute <n_p> (mode occupancies), either projector or unprojected.
# ---------------------------------------------------------------------
#

def get_unprojected_quasiparticle_occupations(spectrum,T):
    """
    this function computes the expectation values of fermionic occupation operators at temperature beta=1/T
    without projection

    it RETURNS: log_partition_function, particle_expectation_values
    """
    beta = _safe_T_to_beta(T)

    if len(spectrum.shape)>1:
        spectrum = np.diag(spectrum)

    # get positive half of spectrum, in descending order
    spectrum = np.abs(spectrum[spectrum.size//2:])[::-1]

    # the log expectation values for the particles normalised by the partition function
    # for a single mode: np.exp( (-beta*spectrum[m] - np.logaddexp( - beta*spectrum[m], beta*spectrum[m] )) )
    log_particle_expectation_values = (-beta*spectrum/2.-np.logaddexp(-beta*spectrum/2.,beta*spectrum/2.))

    # for consistency with the projected method below we also calculate the log_partition function
    return np.exp(log_particle_expectation_values),{'log_Z':get_unprojected_log_Z(spectrum,T)}

def get_projected_quasiparticle_occupations(spectrum,T,desired_parity):
    """
    this function computes the expectation values of fermionic occupation operators at temperature beta=1/T
    only summing over states in the correct parity subspace

    the calculation is similar to the recursive computation of the log_partition_function (which this function
    also calculates for free)

    it RETURNS: log_partition_function, particle_expectation_values
    """
    beta = _safe_T_to_beta(T)

    if len(spectrum.shape)>1:
        spectrum = np.diag(spectrum)

    # we need to include log(0) at some points in these computations, numerically we use -1*(some big number)
    # to achieve this
    _big_num = (1E16)

    # empty numpy 1d arrays to hold the projected fermionic expectation values
    number_modes = spectrum.size//2
    log_particle_one_mode_expct_vals_even = np.zeros(number_modes)
    log_particle_one_mode_expct_vals_odd = np.zeros(number_modes)

    # start from the first mode
    logz_even = beta*spectrum[-1]/2.
    logz_odd = -beta*spectrum[-1]/2.

    # first mode
    # -----------
    # single mode,
    # ------------
    log_particle_one_mode_expct_vals_even[0] = -_big_num
    log_particle_one_mode_expct_vals_odd[0] = -beta*spectrum[-1]/2.
     
    # iterate through the rest of the modes
    # (accessing the +ve eigenvalues in the second half of the spectrum list)
    for mode_p in range(1,number_modes):
        
        # compute the next level of the iteration
        tmp = logz_even, logz_odd
        logz_even = np.logaddexp( tmp[0] + beta*spectrum[-mode_p-1]/2., tmp[1] - beta*spectrum[-mode_p-1]/2. )
        logz_odd = np.logaddexp( tmp[1] + beta*spectrum[-mode_p-1]/2., tmp[0] - beta*spectrum[-mode_p-1]/2. )

        # -----------
        # single mode,
        # occupation prob for mode_p first become different from logZ here
        # ------------
        log_particle_one_mode_expct_vals_even[mode_p] = np.logaddexp( tmp[0] - _big_num, tmp[1] - beta*spectrum[-mode_p-1]/2. )
        log_particle_one_mode_expct_vals_odd[mode_p] = np.logaddexp( tmp[1] - _big_num, tmp[0] - beta*spectrum[-mode_p-1]/2.  )

        # -----------
        # single mode,
        # for modes<mode_p, add the sum over mode_p's marginal probabilities i.e. trace it out
        # ------------
        tmp = copy.deepcopy(log_particle_one_mode_expct_vals_even),\
                copy.deepcopy(log_particle_one_mode_expct_vals_odd)
        log_particle_one_mode_expct_vals_even[:mode_p] = np.logaddexp( tmp[0][:mode_p] + beta*spectrum[-mode_p-1]/2., tmp[1][:mode_p] - beta*spectrum[-mode_p-1]/2. )
        log_particle_one_mode_expct_vals_odd[:mode_p] = np.logaddexp( tmp[1][:mode_p] + beta*spectrum[-mode_p-1]/2., tmp[0][:mode_p] - beta*spectrum[-mode_p-1]/2. )

    # subtract the partition functions from each
    log_particle_one_mode_expct_vals_even = log_particle_one_mode_expct_vals_even - logz_even
    log_particle_one_mode_expct_vals_odd = log_particle_one_mode_expct_vals_odd - logz_odd

    # return the desired parity
    if np.rint(desired_parity)==1:
        return np.exp(log_particle_one_mode_expct_vals_even),{'log_Z':logz_even}
    else:
        return np.exp(log_particle_one_mode_expct_vals_odd),{'log_Z':logz_odd}

#
# ---------------------------------------------------------
# compute <H> (energy), either projector or unprojected.
# ---------------------------------------------------------
#

def get_unprojected_thermal_energy(spectrum,T):
    """
    in the unprojected case:
    ------------------------
    the thermal energy of the two-mode sys (-\en_k,+\en_k) at temperature 
    beta is given by: E_{w}(\beta) = - \sum_k (E_k/2) tanh( \beta E_k/2 ).
    """ 
    beta = _safe_T_to_beta(T)

    if len(spectrum.shape)>1:
        spectrum = np.diag(spectrum)

    # get positive half of spectrum, in descending order
    spectrum = np.abs(spectrum[spectrum.size//2:])

    # compute thermal energy
    E = np.sum(-1.*(spectrum/2.0)*np.tanh(beta*spectrum/2.))

    # get logZ and quasiparticle occupations, to be consistent with the return type of get_projected_thermal_energy
    logz,quasi_occs = get_unprojected_quasiparticle_occupations(spectrum,T)

    return E,{'log_Z':logz,'quasi_occs':quasi_occs}

def get_projected_thermal_energy(spectrum,T,desired_parity):
    """
    in the projected case:
    ----------------------
    we compute the statistical thermal energy \sum_k \en_k ( n_k - 1/2) 
    directly from the particle expct vals
    """
    beta = _safe_T_to_beta(T)

    log_z,particle_expectation_values = get_projected_quasiparticle_occupations(spectrum,T,desired_parity)
    return np.sum(spectrum*(particle_expectation_values-0.5)),{'log_Z':log_z,'quasi_occs':particle_expectation_values}

# 
# ------------------------------------------------------
# compute mode covariances, needed for specific heats
# ------------------------------------------------------
# 

def _compute_projected_langle_np_nq_rangle(spectrum,T,desired_parity):
    """ 
    here we compute < n_k n_l > occupation correlation eigenvalues, we compute for l > k and then
    symmetrise
    """
    beta = _safe_T_to_beta(T)

    if len(spectrum.shape)>1:
        spectrum = np.diag(spectrum)
        
    # we need to include log(0) at some points in these computations, numerically we use -1*(some big number)
    # to achieve this
    _big_num = (1E16)

    # empty numpy 1d arrays to hold the projected fermionic expectation values
    number_modes = spectrum.size//2
    log_particle_one_mode_expct_vals_even = np.zeros(number_modes)
    log_particle_one_mode_expct_vals_odd = np.zeros(number_modes)

    # empty numpy 2d arrays to hold the projected two-mode occupation expectation values
    log_particle_two_mode_expct_vals_even = np.zeros((number_modes,number_modes))
    log_particle_two_mode_expct_vals_odd = np.zeros((number_modes,number_modes))

    # start from the first mode
    logz_even = beta*spectrum[-1]/2.
    logz_odd = -beta*spectrum[-1]/2.

    # first mode
    # -----------
    # single mode,
    # ------------
    log_particle_one_mode_expct_vals_even[0] = -_big_num
    log_particle_one_mode_expct_vals_odd[0] = -beta*spectrum[-1]/2.
    # ------------
    # two mode,
    # insert the occupation probability for mode 0 into column 0; row=1:,col=0
    # ------------
    # prob that mode q (column index) is occupied
    log_particle_two_mode_expct_vals_even[1:,0] = -_big_num
    log_particle_two_mode_expct_vals_odd[1:,0] = -beta*spectrum[-1]/2.
     
    # iterate through the rest of the modes
    # (accessing the +ve eigenvalues in the second half of the spectrum list)
    for mode_p in range(1,number_modes):
        
        # compute the next level of the iteration
        tmp = logz_even, logz_odd
        logz_even = np.logaddexp( tmp[0] + beta*spectrum[-mode_p-1]/2., tmp[1] - beta*spectrum[-mode_p-1]/2. )
        logz_odd = np.logaddexp( tmp[1] + beta*spectrum[-mode_p-1]/2., tmp[0] - beta*spectrum[-mode_p-1]/2. )

        # -----------
        # single mode,
        # occupation prob for mode_p first become different from logZ here
        # ------------
        log_particle_one_mode_expct_vals_even[mode_p] = np.logaddexp( tmp[0] - _big_num, tmp[1] - beta*spectrum[-mode_p-1]/2. )
        log_particle_one_mode_expct_vals_odd[mode_p] = np.logaddexp( tmp[1] - _big_num, tmp[0] - beta*spectrum[-mode_p-1]/2.  )
        # ------------
        # two mode,
        # insert the occupation prob for mode_p into column mode_p;
        # row=(mode_p+1):,col=mode_p i.e. below the diagonal
        # ------------
        log_particle_two_mode_expct_vals_even[(mode_p+1):,mode_p] = np.logaddexp( tmp[0] - _big_num, tmp[1] - beta*spectrum[-mode_p-1]/2. )
        log_particle_two_mode_expct_vals_odd[(mode_p+1):,mode_p] = np.logaddexp( tmp[1] - _big_num, tmp[0] - beta*spectrum[-mode_p-1]/2. )

        # -----------
        # single mode,
        # for modes<mode_p, add the sum over mode_p's marginal probabilities i.e. trace it out
        # ------------
        tmp = copy.deepcopy(log_particle_one_mode_expct_vals_even),\
                copy.deepcopy(log_particle_one_mode_expct_vals_odd)
        log_particle_one_mode_expct_vals_even[:mode_p] = np.logaddexp( tmp[0][:mode_p] + beta*spectrum[-mode_p-1]/2., tmp[1][:mode_p] - beta*spectrum[-mode_p-1]/2. )
        log_particle_one_mode_expct_vals_odd[:mode_p] = np.logaddexp( tmp[1][:mode_p] + beta*spectrum[-mode_p-1]/2., tmp[0][:mode_p] - beta*spectrum[-mode_p-1]/2. )
        # ------------
        # two mode,
        # add the occupation probabilites for mode_p to the values in row mode_p;
        # row=mode_p,col=:mode_p i.e. up to diagonal
        # these will already include the occ prob for mode col
        # ------------
        tmp = copy.deepcopy(log_particle_two_mode_expct_vals_even),\
                copy.deepcopy(log_particle_two_mode_expct_vals_odd)
        log_particle_two_mode_expct_vals_even[mode_p,:mode_p] = np.logaddexp( tmp[0][mode_p,:mode_p] - _big_num, tmp[1][mode_p,:mode_p] - beta*spectrum[-mode_p-1]/2. )
        log_particle_two_mode_expct_vals_odd[mode_p,:mode_p] = np.logaddexp( tmp[1][mode_p,:mode_p] - _big_num, tmp[0][mode_p,:mode_p] - beta*spectrum[-mode_p-1]/2. )

        # ------------
        # two mode,
        # for all rows apart from row=mode_p, add the sum over mode_p's marginal
        # probabilities i.e. trace it out
        # even though we only want the elements below the diagonal we can do it
        # for all cols, and throw away the parts above diag
        # ------------
        all_modes_but_p = np.arange(number_modes)[np.arange(number_modes)!=mode_p]
        log_particle_two_mode_expct_vals_even[np.ix_(all_modes_but_p,all_modes_but_p)] = np.logaddexp( tmp[0][np.ix_(all_modes_but_p,all_modes_but_p)] + beta*spectrum[-mode_p-1]/2., tmp[1][np.ix_(all_modes_but_p,all_modes_but_p)] - beta*spectrum[-mode_p-1]/2. )
        log_particle_two_mode_expct_vals_odd[np.ix_(all_modes_but_p,all_modes_but_p)] = np.logaddexp( tmp[1][np.ix_(all_modes_but_p,all_modes_but_p)] + beta*spectrum[-mode_p-1]/2., tmp[0][np.ix_(all_modes_but_p,all_modes_but_p)] - beta*spectrum[-mode_p-1]/2. )
        log_particle_two_mode_expct_vals_even = np.tril(log_particle_two_mode_expct_vals_even,-1)
        log_particle_two_mode_expct_vals_odd = np.tril(log_particle_two_mode_expct_vals_odd,-1)
        """
        for prev_row in np.arange(mode_p):
            log_particle_two_mode_expct_vals_even[:,:,prev_row,:prev_row] = np.logaddexp( tmp[0][:,:,prev_row,:prev_row] + beta*spectrum[-mode_p-1]/2., tmp[1][:,:,prev_row,:prev_row] - beta*spectrum[-mode_p-1]/2. )
            log_particle_two_mode_expct_vals_odd[:,:,prev_row,:prev_row] = np.logaddexp( tmp[1][:,:,prev_row,:prev_row] + beta*spectrum[-mode_p-1]/2., tmp[0][:,:,prev_row,:prev_row] - beta*spectrum[-mode_p-1]/2. )
        for following_rows in range(mode_p+1,number_modes):
            log_particle_two_mode_expct_vals_even[:,:,following_rows,:mode_p] = np.logaddexp( tmp[0][:,:,following_rows,:mode_p] + beta*spectrum[-mode_p-1]/2., tmp[1][:,:,following_rows,:mode_p] - beta*spectrum[-mode_p-1]/2. )
            log_particle_two_mode_expct_vals_odd[:,:,following_rows,:mode_p] = np.logaddexp( tmp[1][:,:,following_rows,:mode_p] + beta*spectrum[-mode_p-1]/2., tmp[0][:,:,following_rows,:mode_p] - beta*spectrum[-mode_p-1]/2. )
        """

    # symmetrise the mode-wise off-diag elements
    sym_with = np.transpose(np.tril(log_particle_two_mode_expct_vals_even,-1))
    log_particle_two_mode_expct_vals_even = log_particle_two_mode_expct_vals_even + sym_with
    sym_with = np.transpose(np.tril(log_particle_two_mode_expct_vals_odd,-1))
    log_particle_two_mode_expct_vals_odd = log_particle_two_mode_expct_vals_odd + sym_with

    # subtract the partition functions from each
    log_particle_one_mode_expct_vals_even = log_particle_one_mode_expct_vals_even - logz_even
    log_particle_one_mode_expct_vals_odd = log_particle_one_mode_expct_vals_odd - logz_odd
    # normalise two-mode probs with partition func
    log_particle_two_mode_expct_vals_even = log_particle_two_mode_expct_vals_even - logz_even
    log_particle_two_mode_expct_vals_odd = log_particle_two_mode_expct_vals_odd - logz_odd

    # return the desired parity
    if np.rint(desired_parity)==1:
        return logz_even,log_particle_one_mode_expct_vals_even,log_particle_two_mode_expct_vals_even
    else:
        return logz_odd,log_particle_one_mode_expct_vals_odd,log_particle_two_mode_expct_vals_odd

def get_projected_covariance_matrix(spectrum,T,desired_parity):
    """ 
    here we compute the full covariance matrix Cov(n_p,n_q) for the particle expct values.
    from this we get logZ and the particle expectation values for free

    returns: logZ, <n_p>, Cov(n_p,n_q)
    """
    beta = _safe_T_to_beta(T)

    if len(spectrum.shape)>1:
        spectrum = np.diag(spectrum)

    number_modes = spectrum.size//2
    logz,log_particle_one_mode_expct_vals,log_particle_two_mode_expct_vals = _compute_projected_langle_np_nq_rangle(spectrum,T,desired_parity)
    particle_one_mode_expct_vals = np.exp(log_particle_one_mode_expct_vals)
    expt_nn = np.exp(log_particle_two_mode_expct_vals)

    # diag elements of <n_p n_p> = <n_p>
    expt_nn = expt_nn + np.diag(particle_one_mode_expct_vals)

    # <n_p><n_q>
    exptn_exptn = np.outer(particle_one_mode_expct_vals,particle_one_mode_expct_vals)

    # covariance matrix = <n_p n_q> - <n_p><n_q>
    mode_occ_covariance_mat = expt_nn - exptn_exptn

    return mode_occ_covariance_mat,{'log_Z':logz,'quasi_occs':np.exp(log_particle_one_mode_expct_vals)}

# 
# -------------------------------------------------
# compute fermionic correlation matrix <c_i c_j>
# -------------------------------------------------
# 

def _compute_correlation_matrix(quasi_occs,U):
    """ """

    # symmetrise the quasiparticle occupations so it contains both the particle
    # and (redundant) hole information to match the U matrix
    full_quasi_occs = np.zeros(2*quasi_occs.size)
    full_quasi_occs[:quasi_occs.size] = 1-quasi_occs # holes
    full_quasi_occs[quasi_occs.size:] = quasi_occs[::-1] # particles, in order of increasing energy
    full_quasi_occs = np.diag(full_quasi_occs)

    return np.dot(U,np.dot(full_quasi_occs,U.conj().T))

def get_projected_correlation_matrix(D,U,T,desired_parity):
    """ """

    # convert D matrix to spectrum
    spectrum = np.real(1j*D)

    # get the quasiparticle occupations
    quasi_occs,extras = get_projected_quasiparticle_occupations(spectrum,T,desired_parity)
    extras['quasi_occs'] = quasi_occs

    return _compute_correlation_matrix(quasi_occs,U),extras

def get_unprojected_correlation_matrix(D,U,T):
    """ """

    # convert D matrix to spectrum
    spectrum = np.real(1j*D)

    # get the quasiparticle occupations
    quasi_occs,extras = get_unprojected_quasiparticle_occupations(spectrum,T)
    extras['quasi_occs'] = quasi_occs

    return _compute_correlation_matrix(quasi_occs,U),extras

