import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import flylib as flb
import time
import itertools
import sys
import plotting_utls as pltuls
import util


def initialize_W(N,D,T,scale=1):
    return np.random.uniform(0,scale,size=(N,D,T))

def initialize_c(S,N,scale=1):
    return np.random.uniform(0,scale,size=(S,N))


def compute_phi_s_i(M,W,t,i,s):
    # tao = np.range(T)
    T = np.shape(W)[2]
    summand = 0
    #want to shift W by t according to convention specified in 3.1 (i)
    W_t = shift_matrix_columns(t,W[i,:,:])

    for tao in range(T):
        summand += np.dot(M[s,:,tao],W_t[:,tao]) #synergies W are indexed by i = 1,2,...,N muscles j = 1,2,...D column is the time

    if summand<0:
        print(summand)
        print(np.sum(M<0))
        print(np.sum(W_t<0))
        sys.exit()

    return summand

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

def shift_matrix_columns_2(column_shift,A,new_cols):
    A_copy = np.copy(A)
    n_rows,n_cols = np.shape(A_copy)
    padding = np.zeros((n_rows,new_cols))
    double_padded = np.hstack((padding,A_copy,padding))
    starting_index = (new_cols) + (-1)*column_shift
    ending_index = starting_index+new_cols
    to_return = double_padded[:,starting_index:ending_index]
    return to_return

def update_delay(M,W,c,S):
    N,D,T = np.shape(W)
    delays = np.zeros((S,N)) #size of delays (t) is S x N
    for s in range(S):
        M_copy = np.copy(M)
        synergy_list = list(range(N))
        for synergy_step in range(N):
            phi_s_is = np.zeros((len(synergy_list),2*T+1))
            ts = range(-T,T+1)
            for i,synergy in enumerate(synergy_list):
                for delay_index,t in enumerate(ts):
                    phi_s_is[i,delay_index] = compute_phi_s_i(M_copy,W,t,synergy,s)

            max_synergy_index,max_delay_index = np.unravel_index(np.argmax(phi_s_is),np.shape(phi_s_is))
            max_synergy = synergy_list[max_synergy_index]
            max_delay = range(-T,T+1)[max_delay_index]
            # print('max synergy and delay: ',max_synergy,max_delay)
            # raw_input('next?')
            shifted_max_synergy = shift_matrix_columns(max_delay,W[max_synergy,:,:])
            #Original scaling attempt
            scaled_shifted_max_synergy = c[s,max_synergy]*shifted_max_synergy
            #Try a version where the scaling factor matches the peak of the synergy with the peak of M
            # scaled_shifted_max_synergy = (np.max(shifted_max_synergy))/(np.max(
            #     M_copy[s,:,:]))*shifted_max_synergy
            # plt.figure(500)
            # plt.plot(scaled_shifted_max_synergy[1,:])
            # plt.plot(M_copy[s,1,:],'r')
            # raw_input(' ')
            M_copy[s,:,:] -= scaled_shifted_max_synergy
            #This is the piece where we assume we make M nonnegative
            M_copy[M_copy<0] =0.
            synergy_list.remove(max_synergy)
            delays[s,max_synergy] = max_delay
    return delays.astype(int)

def update_c(c_copy,M,W,delays):
    S,N = np.shape(c)
    # mu_c = 1e-3
    c_copy = np.copy(c)
    mu_c = 1e-1
    # c_copy = np.copy(c)
    for s in range(S):
        delta_c = -mu_c*squared_error_gradient_episode(M[s,:,:],c[s,:],W,delays[s,:])
        #Try a rule where entries are only updated if they are not sent to 0
        change_indices = (c_copy[s,:]+delta_c)>0
        # print(change_indices)
        # c_copy[s,change_indices] += delta_c[change_indices]

        #Original version
        c_copy[s,:] += delta_c
        print('delta_c'+str(delta_c))
    #Nonnegativity constraint
    c_copy[c_copy<0] = 0.
    return c_copy

def update_W(c,M,W_copy,delays):
    N,D,T = np.shape(W)
    mu_W = 1e-1
    W_copy = np.copy(W)
    delta_W = np.zeros_like(W_copy)
    for i in range(N):
        for tao in range(T):
            delta_W[i,:,tao] = -mu_W*squared_error_gradient_total(M,c,W,delays,i,tao)
            # delta_w_i_tao = -mu_W*squared_error_gradient_total(M,c,W,delays,i,tao)
            # W_copy[i,:,tao] += delta_w_i_tao

    #--------This piece is for visualizing what delta W looks like--------------

    # plt.figure(100)
    # vmin=np.min(delta_W);vmax=np.max(delta_W)
    # mdpt= 1 - vmax / (vmax + abs(vmin))
    # # cmap = pltuls.shiftedColorMap(matplotlib.cm.RdYlBu_r, start=0, midpoint=mdpt, stop=1.0, name='shiftedcmap')
    # cmap = matplotlib.cm.RdYlBu
    # for i in range(N):
    #     plt.subplot(N,1,i+1)
    #     try:
    #         im = plt.gca().get_images()[0]
    #         im.set_data(delta_W[i,:,:])
    #         im.set_clim(vmin=vmin,vmax=vmax)
    #     except(IndexError):
    #         plt.imshow(delta_W[i,:,:],interpolation='none',cmap=cmap,vmin=vmin,vmax=vmax,aspect=0.2*T/D)
    #         plt.colorbar()

    #------------------

    # raw_input(' ')
    # plt.pause(.3)
    #Try a rule where entries are only updated if they are not sent to 0
    change_indices = (W_copy+delta_W)>0
    W_copy[change_indices] += delta_W[change_indices]
    #Original version
    # W_copy += delta_W
    W_copy[W_copy<0] = 0.
    return W_copy


def squared_error_gradient_episode(M_s,c_s,W,delays_s):
    #Computes the gradient of E_s^2 wrt entries of c_s--
    #(each dimension of c_s is a syngergy coefficent)
    #Inputs:
    # M_s: D x T muscle activity for that episode
    # c_s: N synergy coefficents
    # W : N x D x T synergies
    # delay_s : N delays
    D,T = np.shape(M_s)
    nabla_c = np.zeros_like(c_s)
    N = len(c_s)
    for j in range(N):
        nabla_j_ts = np.zeros(T)
        for t in range(T):
            entries_by_d = 2*(M_s[:,t]-np.sum(W[:,:,t]*c_s[:,None],axis=0))*(
                -1*(shift_matrix_columns(delays_s[j],W[j,:,:])[:,t]))
            nabla_j_ts[t] = np.sum(entries_by_d)
        nabla_c[j] = np.sum(nabla_j_ts)
    return nabla_c

def squared_error_gradient_total(M,c,W,delays,i,tao):

    #Computes the gradient of (sum_s) E^2 wrt entries of w_i_tao--
    #(each dimension of w_i_tao is a muscle activity value)
    #Inputs:
    # M: S x D x T muscle activity
    # c: S x N synergy coefficents
    # W : N x D x T synergies
    # delay : S x N delays

    S,D,T = np.shape(M)
    nabla_W= np.zeros(D)
    _,N = np.shape(delays)

    #The two sections immediately below should be made more efficient with object implementation

    #first, initialize a shifted W so we don't have to compute it again each time
    #produce a matrix S x N x D x T of the Ws for each s,j shift pair
    shifted_W = np.array([[shift_matrix_columns(delays[s,j],W[j,:,:]) for j in range(
        N)] for s in range(S)])
    #for every s,j pair scale the shifted W
    scaled_shifted_W = c[:,:,None,None]*shifted_W
    #then sum across each j
    summed_scaled_shifted_W = np.sum(scaled_shifted_W,axis=1)
    #make a tao/i indicator matrix of shape S x T
    delta_term = np.array([(tao==t-delays[:,i]).astype(int) for t in range(T)]).T
    c_si_delta_term = c[:,i][:,None]*delta_term
    # print(np.shape(c_si_delta_term))
    for d_0 in range(D):
        values_by_s_by_t = 2*(M[:,d_0,:] - summed_scaled_shifted_W[:,d_0,:])*(-1)*c_si_delta_term
        value = np.sum(values_by_s_by_t)
        nabla_W[d_0] = value
    # print(nabla_W)
    return nabla_W

def compute_total_sum_squares(M):
    #Returns the total sum of squares (used for R^2 computation) of M
    return np.sum(np.square(M-np.mean(M)))

def compute_squared_error(W,c,t,M):
    #Returns the sum squared error across episodes as defined at the top of section 3
    #Aka residual sum of squares
    S,N = np.shape(c)
    S,D,T = np.shape(M)
    error = np.zeros((S,T))
    for s in range(S):
        for t in range(T):
            entries_by_d = M[s,:,t]-np.sum(W[:,:,t]*c[s,:][:,None],axis=0)
            error[s,t] = np.sum(np.square(entries_by_d))
    return np.sum(error)

def multiplicative_update_W(M,W_est,H,scale=1):
    # if (np.sum(W_est.dot(H).dot(H.T)==0))>0:
    # plt.figure(333)
    # plt.subplot(3,2,1)
    # plt.imshow(W_est.dot(H).dot(H.T),interpolation='none')
    # plt.subplot(3,2,2)
    # plt.imshow(W_est.dot(H).dot(H.T)==0,interpolation='none')
    # plt.subplot(3,2,3)
    # plt.imshow(H.dot(H.T),interpolation='none')
    # plt.subplot(3,2,4)
    # plt.imshow(H.dot(H.T)[:,np.sum(H.dot(H.T),axis=1)==0],interpolation='none')

    #----This piece displays H and its zero rows-------------
    # plt.figure(444)
    # plt.subplot(2,1,1)
    # plt.imshow(H,interpolation='none')
    # H_demo = np.zeros_like(H);H_demo[np.sum(H,axis=1)==0,:]=1
    # plt.subplot(2,1,2)
    # plt.imshow(H_demo,interpolation='none')
    # plt.show()
    # ----------------------

    zeros = (W_est.dot(H).dot(H.T)==0)
    nonzero_indices = np.logical_not(zeros)
    # print(np.sum(np.dot(M,H.T)[zeros]))
    mult_factor = scale*np.dot(M,H.T)/(W_est.dot(H).dot(H.T))
    # raw_input(' ')
    # print(mult_factor)
    # raw_input(' ')
    # print(np.sum(np.isnan(W_est)))
    W_est[nonzero_indices] = W_est[nonzero_indices]*mult_factor[nonzero_indices]
    # print(np.sum(np.isnan(W_est)))
    return W_est

def multiplicative_update_c(c_est,M,W_est,Theta,H,delays,scale=1):
    _,_,_,T = np.shape(Theta)
    S,N = np.shape(c_est)
    M = util.stack(M,T)
    # plt.figure(9)
    # plt.imshow(H)
    # print(np.shape(H),(N*T,T*S))
    H = util.stack(H,T)
    # print(np.shape(H),(S,N*T,T))
    # plt.figure(10)
    # plt.imshow(H[0,:,:])
    # plt.show()
    mult_factor = np.zeros_like(c_est)
    for s in range(S):
        for i in range(N):
            # print(delays[s,i])
            Theta_i_tis = Theta[i,util.t_shift_to_index(delays[s,i],T),:,:]
            # print(np.sum(Theta_i_tis))
            # raw_input(' ')
            num = np.trace((M[s,:,:].T).dot(W_est).dot(Theta_i_tis))
            denom = np.trace((H[s,:,:].T).dot(W_est.T).dot(W_est).dot(Theta_i_tis))
            # if denom==0:
            #     print(s,i)
            #     print('inf denom c update')
            #     plt.figure(55)
            #     plt.subplot(2,1,1)
            #     plt.imshow((H[s,:,:].T).dot(W_est.T).dot(W_est).dot(Theta_i_tis),
            #     interpolation='none')
            #     plt.subplot(2,1,2)
            #     plt.imshow((H[s,:,:]),interpolation='none')
            #     plt.show()
            #     raw_input(' ')
            mult_factor[s,i] = num/denom
    # print(mult_factor)
    c_est = c_est*scale*mult_factor
    return c_est


def compute_error_by_trace(M,W,C):
    diff = M-W.dot(C)
    return np.trace(np.dot(diff.T,diff))

def match_synergy_estimates_td(W,W_est):
    N,_,_ = np.shape(W_est)
    true_synergies = range(N)
    est_synergies = range(N)
    true_syn_partners = np.zeros(N)
    matching_scores = np.array(
    [[np.sum(np.abs(
        util.normalize(W[true_synergy,:,:])-
            util.normalize(W_est[est_synergy,:,:])))
         for est_synergy in est_synergies]
         for true_synergy in true_synergies])
    true_partners,est_partners = np.unravel_index(
        np.argsort(matching_scores,axis=None),np.shape(matching_scores))
    partners = np.array([true_partners,est_partners])
    for i in range(N):
        true_syn_partners[partners[0,0]] = partners[1,0]
        cols_to_keep = (partners[1,:]!=partners[1,0]) & (partners[0,:]!=partners[0,0])
        partners = partners[:,cols_to_keep]
    true_syn_partners = true_syn_partners.astype('int')
    return true_syn_partners

def construct_Theta(N,T):
    Theta = np.zeros((N,2*T-1,N*T,T)) #shape of each Theta_i(t) is N*T x T
    for i in range(1,N+1):
        for t in range(1-T,T):
            rows,columns = np.indices((N*T,T))
            to_fill = (rows+1-(i-1)*T)==(columns+1-t)
            to_fill[0:(i-1)*T,:] = 0.
            to_fill[i*T:,:] = 0.
            Theta[i-1,util.t_shift_to_index(t,T),:,:] = to_fill
    return Theta



def match_synergy_estimates_sync(W,W_est):
    _,N = np.shape(W_est)
    true_synergies = range(N)
    est_synergies = range(N)
    true_syn_partners = np.zeros(N)
    matching_scores = np.array(
    [[np.sum(np.abs(
        util.normalize(W[:,true_synergy])-
            util.normalize(W_est[:,est_synergy])))
         for est_synergy in est_synergies]
         for true_synergy in true_synergies])
    true_partners,est_partners = np.unravel_index(
        np.argsort(matching_scores,axis=None),np.shape(matching_scores))
    partners = np.array([true_partners,est_partners])
    for i in range(N):
        true_syn_partners[partners[0,0]] = partners[1,0]
        cols_to_keep = (partners[1,:]!=partners[1,0]) & (partners[0,:]!=partners[0,0])
        partners = partners[:,cols_to_keep]
    true_syn_partners = true_syn_partners.astype('int')
    return true_syn_partners


def mult_update_c_synchronous(M,W_est,C_est):
    num = np.dot(W_est.T,M)
    denom = (W_est.T).dot(W_est).dot(C_est)
    C_est = C_est*(num/denom)
    return C_est

def mult_update_W_synchronous(M,W_est,C_est):
    mult_factor = np.dot(M,C_est.T)/(W_est.dot(C_est).dot(C_est.T))
    W_est = W_est*mult_factor
    return W_est
