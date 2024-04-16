import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#import tensorflow_addons.optimizers.AdamW as AdamW
#from tensorflow.keras.optimizers.experimental import AdamW
import numpy as np
from os.path import dirname, join as pjoin
import scipy
import time
import os
import argparse
from utils import *
from PINN_net import *
import json

parser = argparse.ArgumentParser(description='PINN')
parser.add_argument('--js','--json_name', type=str, default='parameter.json', help='json file name.',required=True)
parser.add_argument('--dev','--device', type=int, default=0, help='cuda device number.',required=False)
# DEVICE='cuda:'
args = parser.parse_args()
# tf.config.threading.set_intra_op_parallelism_threads(1)
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.set_visible_devices(physical_devices[args.dev],'GPU')

# Open json file and load
with open(args.js) as json_file:
    json_data = json.load(json_file)

# save_weight_name = json_data['save_weight_name']
data_name = json_data['data_name']
lr = json_data['lr']
max_iter = json_data['max_iter']
num_col = json_data['num_col']
num_z_test = json_data['num_z_test']
num_r_test = json_data['num_r_test']
r_bnd = json_data['r_bnd']
z_bnd = json_data['z_bnd']
r_test_bnd = json_data['r_test']
z_test_bnd = json_data['z_test']
PDE_CO = json_data['L']
seed_num = json_data['seed']
bnd_num =  json_data['bnd_num']
kind = json_data['kind']
net_type =  json_data['net_type']

# parse the list here
r_bnd = [float(item) for item in r_bnd.split(',')]
z_bnd = [float(item) for item in z_bnd.split(',')]
r_test_bnd = [float(item) for item in r_test_bnd.split(',')]
z_test_bnd = [float(item) for item in z_test_bnd.split(',')]

############################
# Process data
############################
data_dir = 'data'
mat_fname = pjoin(data_dir,data_name)
data = scipy.io.loadmat(mat_fname)

psi_mag = data['p_pe_abs'] 
z_train = data['z']
z_train = np.swapaxes(z_train,0,1)
r_train = data['r']
r_train = np.swapaxes(r_train,0,1)
f=np.squeeze(data['f'])

rz_train= np.empty([0, 2])
p_train=np.empty([0, 1])                                                 
print('r_train', r_train.shape, 'z_train: ', z_train.shape)

# Measured point
for r_ind in range(r_train.shape[0]):
    for z_ind in range(z_train.shape[0]):
        rz_temp=np.concatenate([r_train[r_ind:r_ind+1,:], z_train[z_ind:z_ind+1,:]],axis=1)
        rz_train=np.concatenate([rz_train,rz_temp],axis=0) 
        p_train=np.append(p_train,psi_mag[z_ind:z_ind+1,r_ind:r_ind+1],axis=0)

print('r_col_min: ', r_bnd[0], 'r_col_max: ', r_bnd[1])

# Load Sound speed profile
ssp = np.squeeze(data['SSP_save'])
dep = np.squeeze(data['dep_save'])
c0=1500
k0=tf.convert_to_tensor(2*np.pi*f/c0, dtype=tf.float32)

rz_bnd=np.swapaxes(np.vstack([np.linspace(r_bnd[0],r_bnd[1],bnd_num), np.zeros(bnd_num)]),0,1)

# Create folder and save the training data
# created_fol=f"f_{f}_r_bnd({r_bnd[0]}_{r_bnd[1]})_{r_train.shape[0]}_z_bnd({z_bnd[0]}_{z_bnd[1]})_{z_train.shape[0]}"
created_fol=data_name.replace('.mat','')
while(os.path.exists(created_fol)):
    created_fol=created_fol+'_new'
    
createDirectory(created_fol)
scipy.io.savemat(pjoin(created_fol,f'p_train_f{f}.mat'),{'psi_mag':psi_mag,'r_train':r_train,'z_train':z_train})

############################
# Train the model
############################
set_seed(seed_num) # for weight initialization
agent = Pinn(created_fol,kind,lr)

if net_type=='dnn':
    print('not pretrained')
else:
    print('Load weight:',net_type)
    agent.load_weights(net_type)
agent.train(max_iter,f,rz_train,p_train,num_col,r_bnd,z_bnd,bnd_num,k0,ssp,dep,PDE_CO)

############################
# Test the model
############################
r_test=np.linspace(r_test_bnd[0],r_test_bnd[1],num_r_test)
r_test=r_test[:,np.newaxis]
z_test=np.linspace(z_test_bnd[0],z_test_bnd[1],num_z_test)
z_test=z_test[:,np.newaxis]
rz_test= np.empty([0, 2])

for r_ind in range(r_test.shape[0]):
    for z_ind in range(z_test.shape[0]):
        rz_temp=np.concatenate([r_test[r_ind:r_ind+1,:], z_test[z_ind:z_ind+1,:]],axis=1)
        rz_test=np.concatenate([rz_test,rz_temp],axis=0) 

p_result=agent.predict(rz_test)
scipy.io.savemat(pjoin(created_fol,'p_estimate.mat'),{'p_result':p_result.numpy(),'r_test':r_test,'z_test':z_test})

############################
# Save the parameters in json file
############################

json_parameters_save = {'data_name': data_name,
        'lr': lr,
        'max_iter': max_iter,
        'num_col': num_col,
        'num_z_test': num_z_test,
        'num_r_test': num_r_test,
        'r_bnd': r_bnd,
        'z_bnd': z_bnd,
        'r_test': r_test_bnd,
        'z_test': z_test_bnd,
        'L': PDE_CO,
        'seed': seed_num,
        'bnd_num': bnd_num,
        'net_type': net_type,
        'kind': kind
         }

writejson(created_fol,json_parameters_save)

