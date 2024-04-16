# OceanPINN
This repository contains the codes of paper: [Predicting ocean pressure field with a physics-informed neural network](https://pubs.aip.org/asa/jasa/article/155/3/2037/3271348/Predicting-ocean-pressure-field-with-a-physics)

# Requirements
implemented in cuda 12.2 <br/>
python 3.9.13 <br/>
Tensorflow 2.10.0

# Usage
The workflow is as follows:
1) Make the training data '.mat' file.
2) Set the parameters in parameter.json.
3) Execute the main code.
4) Generate the pressure field with trained PINN.
5) Load the PINN-generated data and plot the results.

1. Main code <br/>
Use main.py or Main_jup.ipynb (Same code) to train PINN. <br/>

2. Code to generate pressure field
Use Test_PINN_v2.ipynb to generate a pressure field with trained PINN.
4. parameter.json
Includes parameters for training OceanPINN. Such as the training data 'mat' file name, learning rate, domain and boundary sizes, and number of collocation points. Each parameter is explained in 'Parameter description.txt'.
5. Example case from the paper
Simulation for the noiseless case is included. Training data is in the data folder and the trained PINN model and estimates are in the 'p_109_noiseless' folder. The generated pressure field can be read using 'Load_PINN_results.m'.


http://oalib.hlsresearch.com/AcousticsToolbox/
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
