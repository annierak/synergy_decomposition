#Object-based implementation of time-dependent and synchronous synergies

import substeps
import util
import numpy as np
import matplotlib
import matplotlib.animation as animate
import matplotlib.pyplot as plt
import plotting_utls as pltuls
import time
import sys
import scipy.linalg
import scipy.optimize


class SynchronousSynergyEstimator(object):
    def __init__(self,D,T,N,M,error_threshold=5e-4):
        self.D = D
        self.T = T
        self.N = N
        self.M = M #M is D x T
        self.SS_tot = np.sum(np.square(M-np.mean(M)))
        self.error_threshold = error_threshold

        self.check_shapes()

        self.C_est = np.random.uniform(0,1,(N,T))
        self.W_est = np.random.uniform(0,1,(D,N))
        self.M_est = np.dot(self.W_est,self.C_est)

    def check_shapes(self):
        if np.shape(self.M) != (self.D,self.T):
            raise Exception('M is not DxT, it is shape '+str(np.shape(self.M)))

    def compute_error_by_trace(self):
        diff = self.M-(self.W_est).dot(self.C_est)
        return np.trace(np.dot(diff.T,diff))

    def compute_oneminusrsq(self):
        SS_res = self.compute_error_by_trace()
        return SS_res/self.SS_tot

    def update_C(self):
        num = np.dot(self.W_est.T,self.M)
        denom = (self.W_est.T).dot(self.W_est).dot(self.C_est)
        self.C_est = self.C_est*(num/denom)

    def update_W(self):
        mult_factor = np.dot(self.M,self.C_est.T)/(
            self.W_est.dot(self.C_est).dot(self.C_est.T))
        self.W_est = self.W_est*mult_factor

class TimeDependentSynergyEstimator(object):

    def __init__(self,S,D,T,N,M,error_threshold=1e-5):
        self.D = D
        self.T = T
        self.N = N
        self.S = S
        self.M_stacked = M #M is S x D x T
        self.M_spread = util.spread(M)
        self.SS_tot = self.compute_total_sum_squares()
        self.error_threshold = error_threshold
        self.initialization_scale = 1.

        self.check_shapes()
        self.W_est_stacked = np.random.uniform(
            0,self.initialization_scale,size=(self.N,self.D,self.T))
        self.W_est_spread = util.spread(self.W_est_stacked)
        self.c_est = np.random.uniform(
            0,self.initialization_scale,size=(self.S,self.N))
        self.delays = np.zeros_like(self.c_est) #******** change this

        self.construct_Theta()
        self.update_H()

        self.M_est_spread = self.W_est_spread.dot(self.H_spread)
        self.M_est_stacked = util.stack(self.M_est_spread,2*self.T)

    def check_shapes(self): #******** this should be more thorough
        if np.shape(self.M_stacked) != (self.S,self.D,2*self.T):
            raise Exception('M_stacked is not SxDxT, it is shape '+str(np.shape(self.M_stacked)))

    def compute_total_sum_squares(self):
        return np.sum(np.square(self.M_stacked-np.mean(self.M_stacked)))

    def compute_squared_error(self):
        #Returns the sum squared error across episodes as defined at the top of section 3
        #Aka residual sum of squares
        #This needs to be phased out--switch to the trace version with M-WH
        # error = np.zeros((self.S,self.T))
        # for s in range(self.S):
        #     for t in range(self.T):
        #         entries_by_d = self.M_stacked[s,:,t]-np.sum(
        #             self.W_est_stacked[:,:,t]*self.c_est[s,:][:,None],axis=0)
        #         error[s,t] = np.sum(np.square(entries_by_d))
        # return np.sum(error)
        diff = self.M_spread-self.W_est_spread.dot(self.H_spread)
        return np.trace(np.dot(diff.T,diff))

    def compute_R2(self):
        SS_res = self.compute_squared_error() #*****change this to have actual shifts
        return  (1. - (SS_res/self.SS_tot))

    def update_M_est(self): #********** this doesn't have t capacities yet
        self.M_est_spread = self.W_est_spread.dot(self.H_spread)
        self.M_est_stacked = util.stack(self.M_est_spread,2*self.T)

    def compute_phi_s_i(self,M_temp,t,i,s,debug=False):
        summand = 0
        #want to shift W by t according to convention specified in 3.1 (i)
        W_t = substeps.shift_matrix_columns_2(t,self.W_est_stacked[i,:,:],2*self.T)
        for tao in range(2*self.T):
            summand += np.dot(M_temp[s,:,tao],W_t[:,tao]) #synergies W are indexed by i = 1,2,...,N muscles j = 1,2,...D column is the time
        if debug&(s==0):
            df = plt.figure(999)
            for d in range(self.D):
                plt.subplot(self.D,1,d+1)
                plt.plot(W_t[d,:],label='shifted W')
                plt.plot(M_temp[s,d,:],label='M')
                plt.ylim([0,1])
            plt.legend()
            plt.text(0.5,1,'CC value: '+str(summand),transform=df.transFigure)
            plt.show()
            plt.pause(.001)
            plt.clf()
        return summand

    def update_delays(self,debug=False,penalty=True):
        for s in range(self.S):
            M_copy = np.copy(self.M_stacked)
            synergy_list = list(range(self.N))
            for synergy_step in range(self.N):
                phi_s_is = np.zeros((len(synergy_list),3*self.T-1))
                ts = range(1-self.T,2*self.T)
                for i,synergy in enumerate(synergy_list):
                    for delay_index,t in enumerate(ts):
                        phi_s_is[i,delay_index] = self.compute_phi_s_i(
                            M_copy,t,synergy,s,debug=debug)
                phi_s_is = util.normalize(phi_s_is)
                penalty_term = np.zeros_like(phi_s_is)
                for i,synergy in enumerate(synergy_list):
                    for delay_index,t in enumerate(ts):
                        penalty_term[i,delay_index] = -2.*np.abs(
                        range(1-self.T,2*self.T)[delay_index])
                penalty_term = penalty_term/(np.abs(np.min(penalty_term)))
                max_synergy_index,max_delay_index = np.unravel_index(
                    np.argmax(phi_s_is+(int(penalty)*penalty_term)),np.shape(phi_s_is))
                max_synergy = synergy_list[max_synergy_index]
                max_delay = range(1-self.T,2*self.T)[max_delay_index]
                shifted_max_synergy = substeps.shift_matrix_columns_2(
                    max_delay,self.W_est_stacked[max_synergy,:,:],2*self.T)
                if debug:
                    plt.figure(999)
                    plt.text(0.5,1.,'max t: '+str(max_delay),transform=plt.gcf().transFigure)
                    for d in range(self.D):
                        plt.subplot(self.D,1,d+1)
                        plt.plot(shifted_max_synergy[d,:],label='shifted W')
                        plt.plot(M_copy[s,d,:],label='M')
                    raw_input(' ')


                #Original scaling attempt
                scaled_shifted_max_synergy = self.c_est[s,max_synergy]*shifted_max_synergy
                M_copy[s,:,:] -= scaled_shifted_max_synergy
                #This is the piece where we assume we make M nonnegative
                M_copy[M_copy<0] =0.
                synergy_list.remove(max_synergy)
                self.delays[s,max_synergy] = int(max_delay)

    def construct_Theta(self):
        N = self.N
        T = self.T
        Theta = np.zeros((N,3*T-1,N*T,2*T)) #shape of each Theta_i(t) is N*T x 2*T
        for i in range(1,N+1):
            for t in range(1-T,2*T):
                rows,columns = np.indices((N*T,2*T))
                to_fill = (rows+1-(i-1)*T)==(columns+1-t)
                to_fill[0:(i-1)*T,:] = 0.
                to_fill[i*T:,:] = 0.
                Theta[i-1,util.t_shift_to_index(t,T),:,:] = to_fill
        self.Theta = Theta

    def update_H(self):
        H = np.zeros((self.S,self.N*self.T,2*self.T))
        for s in range(self.S):
            H[s,:,:] = np.sum(self.c_est[s,:][:,None,None]*\
                np.array([self.Theta[i,util.t_shift_to_index(
                self.delays[s,i], self.T),:,:] for i in range(self.N)]),axis=0)
        self.H_stacked = H
        self.H_spread = np.concatenate(H,axis=1)

    def update_c_est(self,scale=1,regress=False):
        if regress:
            #For regression, shape M to (TxD), for each episode (so S x (TxD))
            #(reshape by going through each muscle and then to next)
            M_flat = np.reshape(self.M_stacked,(self.S,self.T*self.D))
            #Each column of predictor is one synergy, shape (TxD)
            #predictor has N columns
            W_flat = np.reshape(self.W_est_stacked,(self.N,self.T*self.D)).T
            #loop through and do a regression for each episode
            for s in range(self.S):
                self.c_est[s,:],_ = scipy.optimize.nnls(W_flat,M_flat[s,:])
        else:
            mult_factor = np.zeros_like(self.c_est)
            for s in range(self.S):
                for i in range(self.N):
                    Theta_i_tis = self.Theta[i,util.t_shift_to_index(self.delays[s,i],self.T),:,:]
                    num = np.trace((self.M_stacked[s,:,:].T).dot(self.W_est_spread).dot(Theta_i_tis))
                    denom = np.trace((self.H_stacked[s,:,:].T).dot(
                        self.W_est_spread.T).dot(self.W_est_spread).dot(Theta_i_tis))
                    mult_factor[s,i] = num/denom
            self.c_est = self.c_est*scale*mult_factor

    def update_W_est(self,scale=1,normalize=False):
        zeros = (self.W_est_spread.dot(self.H_spread).dot(self.H_spread.T)==0)
        nonzero_indices = np.logical_not(zeros)
        mult_factor = scale*np.dot(self.M_spread,self.H_spread.T)/(
            self.W_est_spread.dot(self.H_spread).dot(self.H_spread.T))
        self.W_est_spread[nonzero_indices] = self.W_est_spread[
            nonzero_indices]*mult_factor[nonzero_indices]
        if normalize:
            self.W_est_spread = util.normalize(self.W_est_spread)
        self.W_est_stacked = util.stack(self.W_est_spread,self.T)
