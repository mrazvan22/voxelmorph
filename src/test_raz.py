"""
Test models for MICCAI 2018 submission of VoxelMorph.
"""

import nibabel as nib
import numpy as np
import pickle

import train_miccai2018
import os
import glob

if __name__ == "__main__":
  """
  model training function
  :param data_dir: folder with npz files for each subject.
  :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
  :param model: either vm1 or vm2 (based on CVPR 2018 paper)
  :param model_dir: the model directory to save to
  :param gpu_id: integer specifying the gpu to use
  :param lr: learning rate
  :param n_iterations: number of training iterations
  :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
  :param steps_per_epoch: frequency with which to save models
  :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
  :param load_model_file: optional h5 model file to initialize with
  :param data_loss: data_loss: 'mse' or 'ncc
  """

  data_dir = '/data/vision/polina/projects/wmh/razvan/gasros_affine_transformed_cropped_32_32_32/test'
  atlas_file = '/data/vision/polina/projects/wmh/incoming/2018_05_15/gasros_files/atlas/caa_flair_in_mni_template_smooth.nii.gz'
  # model =

  atlasData = nib.load(atlas_file).get_data()
  print('atlasData.shape', atlasData.shape)
  np.savez('/data/vision/polina/projects/wmh/razvan/gasros_affine_transformed_cropped_32_32_32/test/atlas.npz', vol=atlasData)

  vols = glob.glob(os.path.join(data_dir, '*.nii.gz'))
  for vol in vols:
    data = nib.load(vol).get_data()
    np.savez('%s.npz' % vol.split('.')[0], vol=data)
    print(vol, data.shape)

