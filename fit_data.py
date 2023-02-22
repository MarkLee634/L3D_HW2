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

'''
python fit_data.py --type 'vox'
python fit_data.py --type 'point'
python fit_data.py --type 'mesh'
'''
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


    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=30, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    

    if gif_save == True:
        for i in range (NUM_VIEWS):
            # Prepare the camera:
            # specify elevation and azimuth angles for each viewpoint as tensors. 
            R, T = pytorch3d.renderer.look_at_view_transform(dist=3.0, elev=30, azim=azim[i])
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T)
            rend_gif = renderer(mesh, cameras=cameras, lights=lights)
            rend_gif = rend_gif.cpu().numpy()[:, ..., :3]
            rend_gif = rend_gif[0]


            rend_list.append(rend_gif)
        

        imageio.mimsave(output_path, rend_list, fps=15)

    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def render_from_pcloud(pcloud, device, gif_save, output_path='/home/mark/course/16825L43D/L3D_HW2/images/X.gif'):
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
  
    
    verts = pcloud
    rgb = torch.ones_like(verts)*0.5
    print(f" shape of verts {verts.shape}, shape of rgb {rgb.shape}, rgb {rgb[0][0]}")

    
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(1,1,0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.detach().cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

    NUM_VIEWS = 36
    # transform the camera around the obj 360 degrees
    azim = torch.linspace(-180, 180, NUM_VIEWS)
    rend_list = []

    if gif_save == True:
        for i in range (NUM_VIEWS):
            # Prepare the camera:
            # specify elevation and azimuth angles for each viewpoint as tensors. 
            R, T = pytorch3d.renderer.look_at_view_transform(dist=1.0, elev=30, azim=azim[i])
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T)
            rend = renderer(point_cloud, cameras=cameras)
            rend = rend.detach().cpu().numpy()[:, ..., :3]
            rend = rend[0]


            rend_list.append(rend)
        
        imageio.mimsave(output_path, rend_list, fps=15)


    

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

    titles = ['Initial prediction', 'Ground truth', 'Optimized prediction', 'Ground truth'
    ]
    for ax, im in zip(grid, [init_pred, label, optimized_pred , label]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(titles.pop(0))
    plt.show()


def add_texture_to_mesh(mesh_gt):

    vertices = mesh_gt.verts_list()
    faces = mesh_gt.faces_list()

    # print(f"mesh_gt {mesh_gt}")

    #convert list to tensor
    vertices = torch.cat(vertices)
    faces = torch.cat(faces)

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    color=[0.7, 0.7, 1]


    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color).to(args.device)  # (1, N_v, 3)

    mesh_gt = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )

    return mesh_gt


def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)

    output_path_gt = '/home/mark/course/16825L43D/L3D_HW2/images/mesh_gt.gif'
    output_path = '/home/mark/course/16825L43D/L3D_HW2/images/mesh_target.gif'

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

        #save the first iteration
        if step == 100:
            # render of the unoptimized src
            new_mesh_src_text = add_texture_to_mesh(new_mesh_src)
            rendered_img_init = utils_vox.render_mesh(new_mesh_src_text, args.device)
            # render of the target
            mesh_tgt_text = add_texture_to_mesh(mesh_tgt)
            rendered_img_target = utils_vox.render_mesh(mesh_tgt_text, args.device, gif_save= True, output_path = output_path_gt)     
    
    # render optimized src

    new_mesh_src_text = add_texture_to_mesh(new_mesh_src)
    rendered_img_opt = utils_vox.render_mesh(new_mesh_src_text, args.device, gif_save = True, output_path = output_path)

    # plot all 4 images 
    grid_plot_pred_label(rendered_img_init, rendered_img_opt, rendered_img_target)
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')

    




def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)

    output_path = '/home/mark/course/16825L43D/L3D_HW2/images/Q13_opt.gif'
    output_path_true = '/home/mark/course/16825L43D/L3D_HW2/images/Q13_gt.gif'

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
            rendered_img_init = render_from_pcloud(pointclouds_src, args.device, gif_save=False)
            # render of the target
            rendered_img_target = render_from_pcloud(pointclouds_tgt, args.device,  gif_save=True, output_path=output_path_true)
            
    
    rendered_img_opt = render_from_pcloud(pointclouds_src, args.device,  gif_save=True, output_path=output_path)
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
        voxel_clipped = torch.sigmoid(voxels_src)
        loss = losses.voxel_loss(voxel_clipped,voxels_tgt)
        # print(f"------------ loss {loss} ------------ ")
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time
        loss_vis = loss.cpu().item()

        #save the first iteration
        if step == 100:
            # render of the unoptimized src
            rendered_img_init = render_mesh_from_voxels(voxels_src[0].detach().cpu().numpy(), args.device, gif_save = False)
            # render of the target
            rendered_img_target = render_mesh_from_voxels(voxels_tgt[0].detach().cpu().numpy(), args.device, gif_save = True)


        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))

    print('Done!')

    # render optimized src
    output_path = '/home/mark/course/16825L43D/L3D_HW2/images/voxel_target.gif'
    rendered_img_opt = render_mesh_from_voxels(voxels_src[0].detach().cpu().numpy(), args.device, gif_save = True, output_path = output_path)

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
