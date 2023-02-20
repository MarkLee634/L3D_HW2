import torch
import numpy as np

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d

	#get 1 batch
	voxel_src = voxel_src[0]
	voxel_tgt = voxel_tgt[0]

	#flatten Width x Height x Depth into 1D array
	voxel_src_flat = voxel_src.flatten()
	voxel_tgt_flat = voxel_tgt.flatten()

	#define loss function
	loss = torch.nn.BCELoss()

	#average BCE loss over all voxels
	number_voxels = voxel_src_flat.shape[0]
	loss_voxel = 0


	for i in range(number_voxels):

		#clamp values between 0 and 1

		loss_voxel += loss(torch.clamp(voxel_src_flat[i],min=0.,max=1.),torch.clamp(voxel_tgt_flat[i],min=0.,max=1.))
		

	loss_voxel = loss_voxel/number_voxels

	return loss_voxel

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	return loss_laplacian