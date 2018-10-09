import numpy as np
import matplotlib.pyplot as plt
import plotting_utls as pltuls
#utility functions for algorithm implementation and testing.

def create_ranges(start, stop, N):
    divisor = N-1
    steps = (1.0/divisor) * (stop - start)
    return steps[None,:]*np.arange(N)[:,None] + start[None,:]

def normalize(A):
    return A/np.max(A)

def spread(A):
    #Turns stacked matrix into adjacent matrix spread for multiplicative update
    #reshape from d x e x T to  e x (dxT)
    B = np.copy(A)
    return np.concatenate(B,axis=1)


def stack(A,T):
    #Inverse of above--turns shape of A from e x d*T to d x e x T
    e,dT = np.shape(A)
    B = np.copy(A)
    return np.array(np.hsplit(B,dT/T))


def t_index_to_shift(index,T):
	#for a given time duration T, switch the index of range(1-T,T) to the shift value
	return int((1-T)+index)
def t_shift_to_index(shift,T):
	#do the reverse of the above
	return int(shift +(T-1))

def shift_matrix_columns(column_shift,A,transpose=False):
    A_copy = np.copy(A)
    if transpose:
        A_copy = A_copy.T
    n_rows,n_cols = np.shape(A_copy)
    # if np.abs(column_shift)>n_cols:
    #     raw_input('here'+str(column_shift))
    # column_shift = np.sign(column_shift)*(np.abs(column_shift)%n_cols)
    padding = np.zeros((n_rows,n_cols))
    double_padded = np.hstack((padding,A_copy,padding))
    starting_index = (n_cols) + (-1)*column_shift
    ending_index = starting_index+n_cols
    to_return = double_padded[:,starting_index:ending_index]
    if transpose:
        to_return = to_return.T
    return to_return


def construct_H(c_est,Theta,delays):
    S,N = np.shape(delays)
    _,_,_,T = np.shape(Theta)
    H = np.zeros((S,N*T,T))
    for s in range(S):
        # print(np.shape(c_est[s,:][:,None,None]))
        # print(np.shape(np.array([Theta[i,t_shift_to_index(delays[s,i],T),:,:] for i in range(N)])))
        H[s,:,:] = np.sum(c_est[s,:][:,None,None]*\
            np.array([Theta[i,t_shift_to_index(
            delays[s,i],T),:,:] for i in range(N)]),axis=0)
        # for i in range(N):
            # plt.figure(44)
            # print(s,i)
            # plt.imshow(Theta[i,t_shift_to_index(
            # delays[s,i],T),:,:],interpolation='none')
            # raw_input('')
            # if np.sum(np.sum(Theta[i,t_shift_to_index(
            # delays[s,i],T),:,:],axis=1)==0)>0:
            #     print('flag')
        # if np.sum(np.sum(H[s,:,:],axis=1)==0)
    #last bit is to turn H from S x N*T x T to N*T x T*S
    # plt.figure(666)
    # plt.imshow(H[np.sum(H,axis=1)==0,:],interpolation='none')
    # plt.show()
    # raw_input(' ')

    H = np.concatenate(H,axis=1)
    return H


def gauss_vector(max_x,mu,sigma):
    #Returns a vector of shape (max_x, shape(mu)=shape(sigma)) whose values are a Gaussian
    #function centered on mu, std sigma
    inputs = create_ranges(np.array([1]),max_x,max_x)
    if len(np.shape(mu))>1:
        inputs = inputs[:,:,None]
    return np.exp(-np.power(inputs - mu[None,:], 2.) / (2 * np.power(sigma[None,:], 2.)))

# b = np.zeros((4,100))
def gauss_vector_demo():
    mus = np.array([[10,20],[10,20],[10,20]])
    sigma = np.array([[10,10],[20,20],[30,30]])
    # mus = np.array([10,20,30])
    # sigma = np.array([10,10,10])
    max_x = 100
    output = gauss_vector(max_x,mus,sigma)
    plt.plot(np.reshape(output,(100,6)))
    # to_plot = gauss_vector(100,np.array([50]),np.array([10]))
    # plt.plot(to_plot)
    plt.show()

def test_Theta(Theta):
    plt.close('all')

    #Test that the shifts are working properly
    #Shape of Theta is (N,2*T-1,N*T,T) ; shape of each Theta_i(t) is N*T x T
    #Make a W
    _,_,_,T = np.shape(Theta)
    N=1
    D = 5
    variances = np.random.uniform(0.5*T/20,T/20,(N,D))
    means = np.random.uniform(1,T,(N,D))
    W = gauss_vector(T,means,variances)

    amp_max = 1
    amplitudes = np.random.uniform(0,amp_max,(1,N,D))
    c_min = 0
    c_max = 1

    W = amplitudes*W
    W = np.moveaxis(W,0,2)
    # print(np.shape(W[0,:,:]))
    # plt.figure(5)
    # plt.imshow(W[0,:,:],interpolation='none')
    # raw_input(' ')
    W_repeat = np.hstack([W[0,:,:],W[0,:,:],W[0,:,:]])
    # plt.figure(6)
    # plt.imshow(W_repeat,interpolation='none')
    # raw_input(' ')


    shifts = np.array([-1,2,4])
    shift_indices = shifts + (T-1)

    plt.figure(7)
    ax = plt.subplot2grid((len(shifts)+1,2),(0,1))
    plt.imshow(W[0,:,:],interpolation='none')
    pltuls.strip_bare(ax)
    # padded_W = np.concatenate([np.zeros((D,T-1)),W,np.zeros((D,T))],axis=1)

    counter = 0
    for shift,index in list(zip(shifts,shift_indices)):
        shift_matrix = Theta[0,index,:,:]
        ax = plt.subplot2grid((len(shifts)+1,2),(counter+1,0))
        pltuls.strip_bare(ax
        )
        plt.imshow(shift_matrix,interpolation='none')
        shifted_W = W_repeat.dot(shift_matrix)
        ax = plt.subplot2grid((len(shifts)+1,2),(counter+1,1))
        plt.imshow(shifted_W,interpolation='none')
        pltuls.strip_bare(ax)
        plt.ylabel(shift,rotation=0)

        counter +=1
    plt.show()
    raw_input(' ')
