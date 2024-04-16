import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
from utils import *
import scipy
import os
from os.path import dirname, join as pjoin

class Helm(Model):
    def __init__(self):
        super(Helm, self).__init__()
        self.h1 = Dense(160)
        self.h2 = Dense(160)
        self.h3 = Dense(160)
        self.h4 = Dense(160)
        self.h5 = Dense(160)
        self.u = Dense(2, activation='linear',
        use_bias=False)         

    def call(self, state):
        x =tf.sin(self.h1(state))
        x = tf.sin(self.h2(x))
        x = tf.sin(self.h3(x))
        x = tf.sin(self.h4(x))
        x = tf.sin(self.h5(x))
        out = self.u(x)
        return out

class Pinn(object):
    def __init__(self, folder_name, kind,lr):
        self.lr = lr
        print('Learning rate:' + str(self.lr))
        self.opt = Adam(learning_rate=self.lr)
        #self.k0 = 2*np.pi*150/1500
        self.helm = Helm()
        self.helm.build(input_shape=(None, 2))
        self.folder_name= folder_name
        self.kind=kind
        #self.km = tf.Variable(tf.zeros(5, dtype=tf.float32), dtype=tf.float32, name='km',trainable='true') 
    
    @tf.function
    def physics_net(self, rz, n,k0,kind):
        r=rz[:,0:1]
        z=rz[:,1:2]
        with tf.GradientTape() as taper2, tf.GradientTape() as tapez2:
            tapez2.watch(z)
            taper2.watch(r)
            with tf.GradientTape() as taper, tf.GradientTape() as tapez:
                taper.watch(r)
                tapez.watch(z)  
                phi = self.helm(tf.concat([r,z],axis=1))
                phi_real=phi[:,0:1]
                phi_imag=phi[:,1:2]
                phi_z =tf.squeeze(tapez.batch_jacobian(phi, z))
                phi_r =tf.squeeze(taper.batch_jacobian(phi, r))
#                 phi_r=taper.gradient(phi,r)
            del taper, tapez
            phi_zz =tf.squeeze(tapez2.batch_jacobian(phi_z, z))
            phi_rr =tf.squeeze(taper2.batch_jacobian(phi_r, r))
            phi_zz_real=phi_zz[:,0:1]
            phi_zz_imag=phi_zz[:,1:2]
            phi_r_real=phi_r[:,0:1]
            phi_r_imag=phi_r[:,1:2]
            phi_rr_real=phi_rr[:,0:1]
            phi_rr_imag=phi_rr[:,1:2]
        

        del tapez2, taper2
        if kind==1:
            phi_loss_real=phi_rr_real+phi_zz_real-2*k0*phi_r_imag+k0*k0*(n*n-1)*phi_real
            phi_loss_imag=phi_rr_imag+phi_zz_imag+2*k0*phi_r_real+k0*k0*(n*n-1)*phi_imag
        elif kind==2:
            phi_loss_real=phi_rr_real+phi_zz_real+2*k0*phi_r_imag+k0*k0*(n*n-1)*phi_real
            phi_loss_imag=phi_rr_imag+phi_zz_imag-2*k0*phi_r_real+k0*k0*(n*n-1)*phi_imag

        return (phi_loss_real)**2+(phi_loss_imag)**2, tf.abs(phi_loss_real), tf.abs(phi_loss_imag)
    
    def save_weights(self, name):
        self.helm.save_weights(name)

    def load_weights(self, name):
        self.helm.load_weights(name)

    def learn(self, rz_col, rz_mea,rz_bnd,p_bnd, p_mea, n, k0,pde_co):
        with tf.GradientTape() as tape: #, tf.GradientTape() as tape2:
            f,f_real,f_imag = self.physics_net(rz_col, n,k0,self.kind)          
            loss_pde= tf.reduce_mean(f*pde_co)
            p = self.helm(rz_mea)
            loss_reconstruct = tf.reduce_mean(tf.abs(tf.square(p[:,0:1])+tf.square(p[:,1:2])-tf.square(p_mea)))
            p2 = self.helm(rz_bnd)
            loss_bnd=tf.reduce_mean(((p2-p_bnd))**2)
            loss = loss_pde + loss_reconstruct + loss_bnd
            loss = loss*10**4
        grads = tape.gradient(loss, self.helm.trainable_variables)   
        self.opt.apply_gradients(zip(grads,  self.helm.trainable_variables))
#         time.sleep(10)
        del tape
        return loss, loss_pde, loss_reconstruct, f, f_real, f_imag
    
    def predict(self, rz):
        p = self.helm(rz)
        return p

    def train(self, max_num,f,rz_train,p_train,rz_col_num,r_bnd,z_bnd,bnd_num,k0,ssp,dep,pde_co):        
        train_loss_history = []
        train_loss_pde_history = []
        train_loss_reconstruct_history = []
        
        # Surface boundary points
        rz_bnd=np.swapaxes(np.vstack([np.linspace(r_bnd[0],r_bnd[1],bnd_num), np.zeros(bnd_num)]),0,1)
        p_bnd=np.zeros((bnd_num,2))

        # Test points per 10000 iterations
        r_num_test=100
        z_num_test=201
        r_test=np.linspace(r_bnd[0], r_bnd[1],r_num_test)
        r_test=r_test[:,np.newaxis]
        z_test=np.linspace(dep[0],dep[-1],z_num_test)
        z_test=z_test[:,np.newaxis]
        print('Water depth: ', dep[-1])
        rz_test= np.empty([0, 2])

        for r_ind in range(r_test.shape[0]):
            for z_ind in range(z_test.shape[0]):
                rz_temp=np.concatenate([r_test[r_ind:r_ind+1,:], z_test[z_ind:z_ind+1,:]],axis=1)
                rz_test=np.concatenate([rz_test,rz_temp],axis=0)
        
        # Collocation points
        r_col_data = np.random.uniform(r_bnd[0], r_bnd[1], [rz_col_num, 1])
        z_col_data = np.random.uniform(z_bnd[0], z_bnd[1], [rz_col_num, 1])
        rz_col = np.concatenate([r_col_data, z_col_data], axis=1)
#         rz_col = np.concatenate((rz_col, rz_train), axis=0)
        
        # Sound speed profile
        ssp_col=np.interp(rz_col[:,1], dep, ssp)
        c0=1500
        n_col=tf.convert_to_tensor(c0/ssp_col, dtype=tf.float32)
        n_col=n_col[:,np.newaxis]
        best_loss=100000000
        start_time = time.time()
        rz_col_num_high=int(rz_col_num/3)
        
        print('rz_col_num_high:',rz_col_num_high)
        
        created_fol=self.folder_name
        
        # Iterates
        for iter in range(int(max_num)):
            rz_col_old=rz_col
            loss, loss_pde, loss_reconstruct,res, res_real, res_imag = self.learn(tf.convert_to_tensor(rz_col, dtype=tf.float32), 
                                                                     tf.convert_to_tensor(rz_train, dtype=tf.float32),
                                                                     tf.convert_to_tensor(rz_bnd, dtype=tf.float32), 
                                                                     tf.convert_to_tensor(p_bnd, dtype=tf.float32),
                                                                     tf.convert_to_tensor(p_train, dtype=tf.float32),n_col,k0,pde_co)                      

            train_loss_history.append([iter, loss.numpy()])
            train_loss_pde_history.append([iter, loss_pde.numpy()])
            train_loss_reconstruct_history.append([iter, loss_reconstruct.numpy()])
            
            ind_retain_real = np.argpartition(res_real, -rz_col_num_high,axis=0)[-rz_col_num_high:]
            ind_retain_imag = np.argpartition(res_imag, -rz_col_num_high,axis=0)[-rz_col_num_high:]
            ind_retain = np.concatenate([ind_retain_real,ind_retain_imag])
            ind_retain = np.unique(ind_retain)
#             ind_retain=np.where(np.logical_or(res_real>tf.reduce_mean(res_real)+4*tf.math.reduce_variance(res_real) , res_imag>tf.reduce_mean(res_imag)+4*tf.math.reduce_variance(res_imag)))[0]
            rz_col_num_new=rz_col_num-ind_retain.shape[0]
#             print(ind_retain.shape)
            rz_col_retain=rz_col[ind_retain,:]

            r_col_data = np.random.uniform(r_bnd[0], r_bnd[1], [rz_col_num_new, 1])
            z_col_data = np.random.uniform(z_bnd[0], z_bnd[1], [rz_col_num_new, 1])
            rz_col_new = np.concatenate([r_col_data, z_col_data], axis=1)
            rz_col = np.concatenate((rz_col_retain, rz_col_new), axis=0)
            ssp_col=np.interp(rz_col[:,1], dep, ssp)
            n_col=tf.convert_to_tensor(c0/ssp_col, dtype=tf.float32)
            n_col=n_col[:,np.newaxis]
            
            if iter%10000==0:              
                elapsed = time.time() - start_time
                print('iter=', iter, 'time=', elapsed,', loss=', loss.numpy(), 'loss_pde=', loss_pde.numpy(), 'loss_reconstruct=', loss_reconstruct.numpy())       
                start_time = time.time()
                print('total=',ind_retain.shape,'real= ', ind_retain_real.shape,'imag=', ind_retain_imag.shape)
#                 print(rz_col_retain[:,1])
                p_result = self.helm(tf.convert_to_tensor(rz_test, dtype=tf.float32))
                ind_iter=int(iter/10000)
                scipy.io.savemat(pjoin(created_fol,f'field_residue_f{f}_{ind_iter}.mat'),{'res':res.numpy(),'res_real':res_real.numpy(),'res_imag':res_imag.numpy(),'rz_old':rz_col_old,'rz_col_retain':rz_col_retain,'rz_col_new':rz_col_new,'p_result':p_result.numpy(),'r_test':r_test,'z_test':z_test})

            if loss<best_loss and iter>max_num-20000:
                self.save_weights(pjoin(created_fol,'helm.h5'))
                best_loss=loss
                
        scipy.io.savemat(pjoin(created_fol,'loss_result.mat'),{'loss': np.array(train_loss_history), 'loss_pde': np.array(train_loss_pde_history), 'loss_reconstruct': np.array(train_loss_reconstruct_history)})
        self.save_weights(pjoin(created_fol,'helm_last.h5'))
