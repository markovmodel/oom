import numpy as np
import scipy.linalg as linalg

from pyemma.msm.estimation import cmatrix, count_states, number_of_states
from pyemma.util.types import ensure_dtraj_list

# compute the two step correlation matrix
def two_step_cmatrix(dtrajs, tau):
    nstates = number_of_states(dtrajs)
    C = np.zeros((nstates, nstates, nstates))

    dtrajs = ensure_dtraj_list(dtrajs)
    for dtraj in dtrajs:
        L = dtraj.shape[0]
        """For each 'middle state j' compute a two-step count matrix"""
        for l in range(L-2*tau):
            i = dtraj[l]
            j = dtraj[l+tau]
            k = dtraj[l+2*tau]
            C[j, i, k] += 1

    return C

# return U,S   A=U*S*V'
def truncated_svd_psd(A,m=np.inf):
    m=min(m,A.shape[0])
    S,U=linalg.schur(A)
    s=np.diag(S)
    tol=A.shape[0]*np.spacing(s.max())
    m=min(m,np.count_nonzero(s>tol))
    idx=(-s).argsort()[:m]
    return U[:,idx],np.diag(s[idx])

# return pinv(A)
def pinv_psd(A,m=np.inf):
    U,S=truncated_svd_psd(A)
    return U.dot(np.diag(1.0/np.diag(S))).dot(U.T)

# return R   A=R*R'
def cholcov(A,m=np.inf):
    U,S=truncated_svd_psd(A,m)
    return U.dot(np.sqrt(S))

# return pinv(R)   A=R*R'
def pinv_cholcov(A,m=np.inf):
    U,S=truncated_svd_psd(A,m)
    return np.diag(1.0/np.sqrt(np.diag(S))).dot(U.T)

# estimate the oom parameters
# dtrajs: list of discrete trajectories, tau: lagtime, order: dimension of the oom
def oom(dtrajs, tau, order):
    dtrajs=ensure_dtraj_list(dtrajs)

    pii=np.maximum(count_states(dtrajs),1e-20).reshape(-1)
    pii/=pii.sum()
    C=cmatrix(dtrajs,tau,sliding=True).toarray()+0.0
    C_mem=two_step_cmatrix(dtrajs,tau)+0.0
    C=C+C.T
    C/=C.sum()
    for i in range(C_mem.shape[0]):
        C_mem[i]=C_mem[i]+C_mem[i].T
    C_mem/=C_mem.sum()
    nstates=pii.shape[0]
    
    D=np.diag(1/np.sqrt(pii))
    pinv_R=pinv_cholcov(D.dot(C).dot(D),order)
    order=pinv_R.shape[0]

    Xi_set=np.empty((nstates,order,order))
    for i in range(C_mem.shape[0]):
        Xi_set[i]=pinv_R.dot(D).dot(C_mem[i]).dot(D).dot(pinv_R.T)
    
    omega=pii.reshape(1,-1).dot(D).dot(pinv_R.T)
    sigma=omega.reshape(-1,1)
    return {'sigma': sigma, 'omega': omega, 'Xi_set': Xi_set}

# return eigenvalues and projected eigenvector matrix Q, where Q[:,i] is the i-th projected eigenvector
def oom_spectral_analysis(oom_dict):
    nstates=oom_dict['Xi_set'].shape[0]
    order=oom_dict['sigma'].shape[0]
    Xi_0=oom_dict['Xi_set'].sum(0)
    v,w=linalg.eig(Xi_0)
    v=np.real(v)
    w=np.real(w)
    
    assert(np.linalg.matrix_rank(Xi_0) == order), 'The sum of all observable operators is singular.'
    assert(np.allclose(w.dot(np.diag(v)).dot(linalg.inv(w)),Xi_0)), 'The sum of all observable operators is not diagonalizable.'
    
    inv_V=linalg.inv(np.diag(v))
    inv_w=linalg.inv(w)
    tmp_Q_1=np.empty((nstates,order))
    tmp_Q_2=np.empty((nstates,order))
    for k in range(nstates):
        tmp_Q_1[k]=(oom_dict['sigma'].T.dot(oom_dict['Xi_set'][k].T).dot(inv_w.T).dot(inv_V)).reshape(-1)
        tmp_Q_2[k]=(oom_dict['omega'].dot(oom_dict['Xi_set'][k]).dot(w)).reshape(-1)
    
    eigen_vectors=0.5*(np.sign(tmp_Q_1)+np.sign(tmp_Q_2))*np.sqrt(np.maximum(tmp_Q_1*tmp_Q_2,0.0));
    idx=(-v).argsort()
    eigen_values=v[idx]
    eigen_vectors=eigen_vectors[:,idx]
    if eigen_vectors[:,0].sum()<0:
        eigen_vectors[:,0]=-eigen_vectors[:,0]

    return eigen_values,eigen_vectors

if __name__=="__main__":
    #a simple example
    
    #generate two Markov chains with transition matrix T
    T=np.diag(np.array([3,2,1]))+0.1
    T=T/T.sum(0)[:,np.newaxis]
    pii=np.ones(3)/3.0
    sim_length=100000
    
    s_mem=np.zeros(sim_length)
    for t in range(sim_length):
        if t==0:
            tmp_p=pii
        else:
            tmp_p=T[s]
        s=np.nonzero(np.random.rand()<np.cumsum(tmp_p))[0][0]
        s_mem[t]=s
    s_mem_1=s_mem

    s_mem=np.zeros(sim_length)
    for t in range(sim_length):
        if t==0:
            tmp_p=pii
        else:
            tmp_p=T[s]
        s=np.nonzero(np.random.rand()<np.cumsum(tmp_p))[0][0]
        s_mem[t]=s
    s_mem_2=s_mem

    #The observation value is randomly s_mem[t]*2 or s_mem[t]*2+1. (There are totally 6 observation values)    
    dtraj_1=s_mem_1*2+np.random.randint(2,size=s_mem_1.shape)
    dtraj_2=s_mem_2*2+np.random.randint(2,size=s_mem_2.shape)
    
    dtrajs = [dtraj_1, dtraj_2]
    tau=1
    order=3

    #perform the oom estimation
    oom_dict=oom(dtrajs, tau, order)
    lambd,Q=oom_spectral_analysis(oom_dict)
    
    #print the eigenvalues and eigenvectors
    print lambd
    print Q
