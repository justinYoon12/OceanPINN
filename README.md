# OceanPINN
This repository contains the codes of paper: [Predicting ocean pressure field with a physics-informed neural network](https://pubs.aip.org/asa/jasa/article/155/3/2037/3271348/Predicting-ocean-pressure-field-with-a-physics)


# Requirements
implemented in cuda 12.2 <br/>
python 3.9.13 <br/>
Tensorflow 2.10.0

# Usage
The workflow is as follows:
1) Make the training data (.mat format).
2) Set the parameters in parameter.json.
3) Execute the main code.
4) Generate the pressure field with trained PINN.
5) Load the PINN-generated data and plot the results.

Details for each procedure: <br/>
**1. Training data** <br/>
format: .mat files <br/>
Include serveral variables:
  * r: Sampled range
  * z: Sampled depth
  * p_pe_abs : Sampled envelope magnitude
  * f: Frequency
  * SSP_save: Sound speed profile
  * dep_save: Corresponding depth for sound speed profile

For simulation in the SWellEx-96 environmet as the paper, you can use codes in the 'Kraken simulation' folder (requires [kraken](http://oalib.hlsresearch.com/AcousticsToolbox/) installed).
The data used in the paper is available in the 'data' folder for reference.

**2. parameter.json**  <br/>
Includes parameters for training OceanPINN:
  * data_name: training data file name (.mat format)
  * lr : learning rate
  * max_iter : maximum number of iterations
  * num_col: number of collocation points
  * bnd_num: number of boundary points
  * r_bnd : lower and upper bound of range (to sample collocation points)
  * z_bnd : lower and upper bound of depth (to sample collocation points)
  * num_r_test : number of range samples in generated field (for test in the last iteration)
  * num_z_test : number of depth samples in generated field (for test in the last iteration)
  * r_test : lower and upper bound of range for generated field after training (for test in the last iteration)
  * z_test : lower and upper bound of depth for generated field after training (for test in the last iteration)
  * L : Loss weight coefficient for PDE loss
  * seed : random seed
  * kind : Hankel First (1) or Second (2) depend on sign of Fourier transform 
  * net_type : If "dnn", it is not pretrained. If you specify the pretrained model name, it is loaded.

**3. Main code** <br/>
Use main.py or Main_jup.ipynb (Same code) to train PINN. <br/>

**4. Code to generate pressure field with trained PINN** <br/>
Use Test_PINN_v2.ipynb to generate a pressure field with trained PINN.<br/>

**5. Example case from the paper** <br/>
Trained PINN model for SWellEx-96 simulation in noiseless condition is included in 'p_109_noiseless' Folder. <br/>
The generated pressure field can be read using 'Load_PINN_results.m'. <br/>

# Citation
```
@article{yoon2024predicting,
  title={Predicting ocean pressure field with a physics-informed neural network},
  author={Yoon, Seunghyun and Park, Yongsung and Gerstoft, Peter and Seong, Woojae},
  journal={The Journal of the Acoustical Society of America},
  volume={155},
  number={3},
  pages={2037--2049},
  year={2024}
}
```
