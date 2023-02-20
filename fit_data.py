import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch

import sys
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import imageio
from mpl_toolkits.axes_grid1 import ImageGrid

import utils_vox

def render_mesh_from_voxels(voxels, device, gif_save, output_path='/home/mark/course/16825L43D/L3D_HW2/images/X.gif'):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """
    image_size=256
    voxel_size = 32

    if device is None:
        device = utils_vox.get_device()

    min_value = -1.1
    max_value = 1.1

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )

    NUM_VIEWS = 36
    # transform the camera around the obj 360 degrees
    azim = torch.linspace(-180, 180, NUM_VIEWS)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.5, -4.0]], device=device)
    renderer = utils_vox.get_mesh_renderer(image_size=image_size, device=device)
    rend_list = []


    R, T = pytorch3d.renderer.look_at_view_transform(dist=10, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    

    if gif_save == True:
        for i in range (NUM_VIEWS):
            # Prepare the camera:
            # specify elevation and azimuth angles for each viewpoint as tensors. 
            R, T = pytorch3d.renderer.look_at_view_transform(dist=5.0, elev=30, azim=azim[i])
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T)
            rend = renderer(mesh, cameras=cameras, lights=lights)
            rend = rend.cpu().numpy()[:, ..., :3]
            rend = rend[0]


            rend_list.append(rend)
        

        imageio.mimsave(output_path, rend_list, fps=15)

    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def render_from_pcloud(pcloud, device):
    """
    Renders a point cloud.
    """
    image_size = 256
    background_color=(1, 1, 1)

    if device is None:
        device = utils_vox.get_device()
    renderer = utils_vox.get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    # ----------------------------------------
    # point_cloud = np.load(point_cloud_path)
    # point_cloud = pcloud
    
    # verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    # rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    # ----------------------------------------

   
    verts = pcloud
    rgb = torch.ones_like(verts)*0.5
    print(f" shape of verts {verts.shape}, shape of rgb {rgb.shape}, rgb {rgb[0][0]}")

    
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4,10,0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.detach().cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

    

    return rend


def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    return parser

def grid_plot_pred_label(init_pred, optimized_pred, label):

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, [init_pred, label, optimized_pred , label]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()

def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))

        #save the first iteration for plot
        if step == 0:
            # render of the unoptimized src
            rendered_img_init = render_from_pcloud(pointclouds_src, args.device)
            # render of the target
            rendered_img_target = render_from_pcloud(pointclouds_tgt, args.device)
            

    rendered_img_opt = render_from_pcloud(pointclouds_src, args.device)
    print('Done!')
    # plot all 4 images 
    grid_plot_pred_label(rendered_img_init, rendered_img_opt, rendered_img_target)


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)


    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        # print(f"------------ voxels_src.shape {voxels_src.shape}, {voxels_tgt.shape} ------------ ")
        loss = losses.voxel_loss(voxels_src,voxels_tgt)
        # print(f"------------ loss {loss} ------------ ")
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time
        loss_vis = loss.cpu().item()

        #save the first iteration
        if step == 0:
            # render of the unoptimized src
            rendered_img_init = render_mesh_from_voxels(voxels_src[0].detach().cpu().numpy(), args.device, gif_save = False)
            # render of the target
            rendered_img_target = render_mesh_from_voxels(voxels_tgt[0].detach().cpu().numpy(), args.device, gif_save = False)


        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))

    print('Done!')

    # render optimized src
    output_path = '/home/mark/course/16825L43D/L3D_HW2/images/voxel_target.gif'
    rendered_img_opt = render_mesh_from_voxels(voxels_src[0].detach().cpu().numpy(), args.device, gif_save = False, output_path = output_path)

    # plot all 4 images 
    grid_plot_pred_label(rendered_img_init, rendered_img_opt, rendered_img_target)



    sys.exit()


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        print(f"------------ pointclouds_src.shape {pointclouds_src.shape}, {pointclouds_tgt.shape} ------------ ")

        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)        
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args)        


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
