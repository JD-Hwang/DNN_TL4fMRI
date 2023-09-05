# DNN_TL4fMRI
  - This is code for comparing the efficiacy of transfer learning using AE (Autoencoder) or RBM (restricted Boltzmann machine), and randomly initialized weights

Please cite this work as follows:

  - Hwang, J., Lustig, N., Jung, M., & Lee, J. H. (2023). Autoencoder and restricted Boltzmann machine for transfer learning in functional magnetic resonance imaging task classification. Heliyon, 9(7).
(https://www.cell.com/heliyon/pdf/S2405-8440(23)05294-5.pdf)

![Figure1](https://github.com/JD-Hwang/DNN_TL4fMRI/assets/65854964/eb2bbd85-168d-4d10-b589-b38a2c555aac)


**Belows are the description of each python file**


  - *TL4fMRI_load_sbj.py*
    -  Includes functions for loading subjects, and concat to train model

- *TL4fMRI_save_data.py*
  -  Includes functions for make directory for result files, parameters, trained models, result plots

- *TL4fMRI_sparsity_control.py*
  -  Includes functions for calculate & control Hoyer's sparsity 
