from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 1 x 32 x 32 x 32
            print(f" ***** create model for {args.type} *****")
            # 5 3D conv layers
            # channel size [512,128,64,32,8,1]
            # kernals size [4x4x4 with stride of 2, padding 1]
            # [ 3D conv > batch > ReLU ] x5 > sigmoid
            in_size = 512
            out_size = 1

            #============================================================
            # layers = [
            #    # [ 3D conv > batch > ReLU ] x5 > sigmoid 
            #    #input layer
            #     nn.Conv3d(in_size, 128, kernel_size=4, stride=2, padding=1),
            #     nn.BatchNorm3d(128),
            #     nn.ReLU(),

            #    #1st layer
            #    #2nd layer
            #     #3rd layer
            #     #4th layer
            #     #5th layer
            #     #output layer
            #     nn.Conv3d(128, out_size, kernel_size=4, stride=2, padding=1)
            # ]

            # self.decoder = nn.Sequential(*layers)
            #============================================================
            # Layer Definition
            self.layer1 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(in_size, 128, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm3d(128),
                torch.nn.ReLU()
            )
            self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2,  padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
            )
            self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2,  padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
            )
            self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2,  padding=1),
            torch.nn.Sigmoid()
            )



            pass
            # TODO:
            # self.decoder =             
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            # self.decoder =             
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.decoder =             

    def forward(self, images, args):
        # print(f" ----------- forward pass -----------")
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            # voxels_pred =  self.decoder(encoded_feat)   
            # 
            # print(f"shape of encoded_feat: {encoded_feat.shape}")
            # reshape to [bx512] to [b x 512 x 1 x 1 x 1]
            encoded_feat_reshaped = encoded_feat.view(-1, 512, 1, 1, 1)

            # print(f"shape of encoded_feat_reshaped: {encoded_feat_reshaped.shape}")

            # upsample to b x 512 x 2 x 2 x 2
            upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
            encoded_feat_upsampled = upsample(encoded_feat_reshaped)

            # print(f"shape of encoded_feat_upsampled: {encoded_feat_upsampled.shape}")

            # forward pass through 5 layers
            voxels_pred = self.layer1(encoded_feat_upsampled)
            # print(f"after layer1: {voxels_pred.shape}")    
            voxels_pred = self.layer2(voxels_pred)
            # print(f"after layer2: {voxels_pred.shape}")  
            voxels_pred = self.layer3(voxels_pred)
            # print(f"after layer3: {voxels_pred.shape}")  
            voxels_pred = self.layer4(voxels_pred)
            # print(f"after layer4: {voxels_pred.shape}")  



            return voxels_pred

        elif args.type == "point":
            # TODO:
            # pointclouds_pred =             
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred =             
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

