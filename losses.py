import torch
import numpy as np
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import chamfer_distance
import sys
import torch.nn.functional 

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d

	# #get 1 batch
	# voxel_src = voxel_src.squeeze(0) 
	# voxel_tgt = voxel_tgt.squeeze(0) 

	# #flatten Width x Height x Depth into 1D array
	# voxel_src_flat = voxel_src.flatten()
	# voxel_tgt_flat = voxel_tgt.flatten()

	# #define loss function
	# loss = torch.nn.BCELoss()

	# #average BCE loss over all voxels
	# number_voxels = voxel_src_flat.shape[0]
	# loss_voxel = 0


	# for i in range(number_voxels):

	# 	#clamp values between 0 and 1
	# 	loss_voxel += loss(torch.clamp(voxel_src_flat[i],min=0.,max=1.),torch.clamp(voxel_tgt_flat[i],min=0.,max=1.))
		
	# #averaged loss
	# loss_voxel = loss_voxel/number_voxels

	loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(voxel_src), voxel_tgt)
	
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch


	# loss_chamfer = 0
	# loss_chamfer_tuple = chamfer_distance(point_cloud_src,point_cloud_tgt) #for validation only
	# loss_chamfer = loss_chamfer_tuple[0]
	# print(f" ******** loss_chamfer {loss_chamfer} ********")

	x_nn = knn_points(point_cloud_src, point_cloud_tgt, K=1)
	y_nn = knn_points(point_cloud_tgt, point_cloud_src, K=1)
	cham_x = x_nn.dists[..., 0]  # (N, P1)
	cham_y = y_nn.dists[..., 0]  # (N, P1)

	# Apply point reduction
	cham_x = cham_x.sum(1)  # (N,)
	cham_y = cham_y.sum(1)  # (N,)

	cham_dist = cham_x + cham_y  # (N,)
	loss_chamfer = cham_dist
	print(f" ******** loss_chamfer {loss_chamfer} ********")

	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	return loss_laplacian