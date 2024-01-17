import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .RevIN import RevIN
from tkinter import _flatten
from einops.layers.torch import Rearrange


from .embed import (
    DataEmbedding,
    PositionalEmbedding,
    ChInd_PositionalEmbedding,
    ChInd_DataEmbedding,
)
from einops import reduce


def get_activation(activ):
    if activ == "relu":
        return nn.ReLU()
    elif activ == "gelu":
        return nn.GELU()
    elif activ == "leaky_relu":
        return nn.LeakyReLU()
    elif activ == "none":
        return nn.Identity()
    else:
        raise ValueError(f"activation:{activ}")

class MLPBlock(nn.Module):
    def __init__(
        self,
        dim,
        in_features: int,
        hid_features: int,
        out_features: int,
        activ="gelu",
        drop: float = 0.00,
        jump_conn="proj",
    ):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hid_features),
            get_activation(activ),
            nn.Linear(hid_features, out_features),
            nn.Dropout(drop),
        )
        if jump_conn == "trunc":
            self.jump_net = nn.Identity()
        elif jump_conn == "proj":
            self.jump_net = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"jump_conn:{jump_conn}")

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = self.jump_net(x) + self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x
    


class PatchMLP_layer(nn.Module):
    def __init__(
        self,
        in_len: int,
        hid_len: int,

        in_chn: int,
        hid_chn: int,
        out_chn,

        patch_size: int,
        hid_pch: int,

        d_model: int,
        norm=None,
        activ="gelu",
        drop: float = 0.0,
        jump_conn='proj'
    ) -> None:
        super().__init__()
        # B C N P
        self.ch_mixing1 = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop, jump_conn=jump_conn)
        # self.ch_mixing2 = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop, jump_conn=jump_conn)
        self.patch_num_mix = MLPBlock(2, in_len // patch_size, hid_len, in_len // patch_size, activ, drop, jump_conn=jump_conn)
        self.patch_size_mix = MLPBlock(2, patch_size, hid_pch, patch_size, activ, drop,jump_conn=jump_conn)
        self.d_mixing1 = MLPBlock(3, d_model, d_model, d_model, activ, drop, jump_conn=jump_conn)

        if norm == 'bn':
            norm_class = nn.BatchNorm2d
        elif norm == 'in':
            norm_class = nn.InstanceNorm2d
        else:
            norm_class = nn.Identity
        self.norm1 = norm_class(in_chn)
        self.norm2 = norm_class(out_chn)

    def forward(self, x_patch_num, x_patch_size):
        # B C N P
        x_patch_num = self.norm1(x_patch_num)
        x_patch_num = self.ch_mixing1(x_patch_num)
        x_patch_num = self.norm2(x_patch_num)
        x_patch_num = self.patch_num_mix(x_patch_num)
        x_patch_num = self.norm2(x_patch_num)
        x_patch_num = self.d_mixing1(x_patch_num)

        x_patch_size = self.norm1(x_patch_size)
        x_patch_size = self.ch_mixing1(x_patch_size)
        x_patch_size = self.norm2(x_patch_size)
        x_patch_size = self.patch_size_mix(x_patch_size)
        x_patch_size = self.norm2(x_patch_size)
        x_patch_size = self.d_mixing1(x_patch_size)

        return x_patch_num, x_patch_size


class Encoder(nn.Module):
    def __init__(self, enc_layers):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList(enc_layers)
        self.num_mix_layer = nn.Sequential(nn.Linear(len(enc_layers), len(enc_layers)*2), nn.Sigmoid(), nn.Linear(len(enc_layers)*2,1), nn.Sigmoid(), Rearrange('b n p k -> b (n p) k'))
        self.size_mix_layer = nn.Sequential(nn.Linear(len(enc_layers), len(enc_layers)*2), nn.Sigmoid(), nn.Linear(len(enc_layers)*2,1) , nn.Sigmoid(), Rearrange('b n p k -> b (n p) k'))
        self.softmax = nn.Softmax(-1)


    def forward(self, x_patch_num, x_patch_size, mask=None):
        num_dist_list = []
        size_dist_list = []
        num_logi_list = []
        size_logi_list = []

        for enc in self.enc_layers:
            x_pach_num_dist, x_patch_size_dist = enc(x_patch_num, x_patch_size)

            num_logi_list.append(x_pach_num_dist.mean(1))
            size_logi_list.append(x_patch_size_dist.mean(1))


            x_pach_num_dist = self.softmax(x_pach_num_dist)
            x_patch_size_dist = self.softmax(x_patch_size_dist)

            x_pach_num_dist = reduce(
                x_pach_num_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            x_patch_size_dist = reduce(
                x_patch_size_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            
            x_pach_num_dist = rearrange(x_pach_num_dist, "b n p -> b (n p) 1")
            x_patch_size_dist = rearrange(x_patch_size_dist, "b p n -> b (p n) 1")

            num_dist_list.append(x_pach_num_dist)
            size_dist_list.append(x_patch_size_dist)

        return num_dist_list, size_dist_list, num_logi_list, size_logi_list
    

class Ensemble_block(nn.Module):
    def __init__(self, e_layers) -> None:
        super().__init__()
        self.mix_layer = nn.parameter.Parameter(torch.ones(e_layers), requires_grad=True)
        pass
    
    def forward(self, dist_list):
        # list of B N D
        dist_list = torch.stack(dist_list, dim=-1)

        # Apply softmax to the mix_layer weights
        weights = torch.softmax(self.mix_layer, dim=0)

        # Apply the weights to dist_list
        dist_list = dist_list * weights

        dist_list = torch.split(dist_list, 1, dim=3)
        dist_list = [t.squeeze(3) for t in dist_list]

        return dist_list

class Mean_Ensemble_block(nn.Module):
    def __init__(self, e_layers) -> None:
        super().__init__()
        pass

    def forward(self, dist_list):
        dist_list = torch.stack(dist_list, dim=-1).mean(-1,keepdim=False)

        return [dist_list]

        


class Encoder_Ensemble(nn.Module):
    def __init__(self, enc_layers ):
        super(Encoder_Ensemble, self).__init__()
        self.enc_layers = nn.ModuleList(enc_layers)
 

        self.num_mix_layer = Ensemble_block(len(enc_layers))
        self.size_mix_layer = Ensemble_block(len(enc_layers))


        self.softmax = nn.Softmax(-1)


    def forward(self, x_patch_num, x_patch_size, mask=None):
        num_dist_list = []
        size_dist_list = []
        num_logi_list = []
        size_logi_list = []

        for enc in self.enc_layers:
            x_pach_num_dist, x_patch_size_dist = enc(x_patch_num, x_patch_size)

            num_logi_list.append(x_pach_num_dist.mean(1))
            size_logi_list.append(x_patch_size_dist.mean(1))


            x_pach_num_dist = self.softmax(x_pach_num_dist)
            x_patch_size_dist = self.softmax(x_patch_size_dist)

            x_pach_num_dist = reduce(
                x_pach_num_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            x_patch_size_dist = reduce(
                x_patch_size_dist,
                "b reduce_c n d -> b n d",
                "mean",
            )
            
            num_dist_list.append(x_pach_num_dist)
            size_dist_list.append(x_patch_size_dist)

            
        num_dist_list = self.num_mix_layer(num_dist_list)
        size_dist_list = self.size_mix_layer(size_dist_list)
        

        return num_dist_list, size_dist_list, num_logi_list, size_logi_list



class PatchMLPAD(nn.Module):
    def __init__(
        self,
        win_size,
        d_model=50,
        expand_ratio=1.2,
        e_layer=3,
        patch_sizes=[2, 4, 8],
        dropout=0.0,
        activation="gelu",
        channel=55,
        cont_model=None,
        norm='in',
        output_attention=True,
    ):
        super(PatchMLPAD, self).__init__()
        self.patch_sizes = patch_sizes
        self.win_size = win_size
        self.output_attention = output_attention

        self.win_emb = PositionalEmbedding(channel)

        self.patch_num_emb = nn.ModuleList(
            [
                nn.Linear(patch_size,d_model) for patch_size in patch_sizes
            ]
        )
        self.patch_size_emb = nn.ModuleList(
            [
                nn.Linear(win_size//patch_size, d_model) for patch_size in patch_sizes
            ]
        )
        self.patch_encoders = nn.ModuleList()
        cont_model = d_model if cont_model is None else cont_model
        cont_model = 30
        if norm == 'bn':
            norm_class = nn.BatchNorm1d
        elif norm == 'in':
            norm_class = nn.InstanceNorm1d
        else:
            norm_class = nn.Identity

        self.patch_num_mixer = nn.Sequential(MLPBlock(2, d_model, d_model//2, d_model, activ='none', drop=dropout, jump_conn='trunc'),nn.Softmax(-1))
        self.patch_size_mixer = nn.Sequential(MLPBlock(2, d_model, d_model//2, d_model, activ='none', drop=dropout, jump_conn='trunc'),nn.Softmax(-1))


        for i, p in enumerate(patch_sizes):
            # Multi patch
            patch_size = patch_sizes[i]
            patch_num = win_size // patch_size
            enc_layers = [
                PatchMLP_layer(win_size, 40, channel, int(channel*1.2), int(channel*1), patch_size, int(patch_size*1.2), d_model, norm, activation, dropout, jump_conn='proj')
                for i in range(e_layer)
            ]
            enc = Encoder_Ensemble(enc_layers=enc_layers)
            self.patch_encoders.append(enc)

    def forward(self, x, mask=None):
        B, L, M = x.shape  # Batch win_size channel
        patch_num_distribution_list = []
        patch_size_distribution_list = []
        patch_num_mx_list = []
        patch_size_mx_list = []
        mask_patch_num_list = []
        mask_patch_size_list = []
        revin_layer = RevIN(num_features=M).to(x.device)

        # Instance Normalization Operation
        x = revin_layer(x, "norm")

        # Mutil-scale Patching Operation
        for patch_index, patchsize in enumerate(self.patch_sizes):
            patch_enc = self.patch_encoders[patch_index]
            x = x + self.win_emb(x)
            # x = self.win_emb(x)
            x_patch_num = x_patch_size = x
            # B L C

            x_patch_num = rearrange(x_patch_num, "b (n p) c -> b c n p", p=patchsize)
            x_patch_size = rearrange(x_patch_size, "b (p n) c-> b c p n", p=patchsize)

            x_patch_num = self.patch_num_emb[patch_index](x_patch_num)
            x_patch_size = self.patch_size_emb[patch_index](x_patch_size)

            # B C N D
            (
                patch_num_distribution,
                patch_size_distribution,
                logi_patch_num,
                logi_patch_size,
            ) = patch_enc(x_patch_num, x_patch_size, mask)

            patch_num_distribution_list.append(patch_num_distribution)
            patch_size_distribution_list.append(patch_size_distribution)

            for i in range(len(logi_patch_num)):
                logi_patch_num1 = logi_patch_num[i]
                logi_patch_size1 = logi_patch_size[i]
                patch_num_mx = self.patch_num_mixer(logi_patch_num1)
                patch_size_mx = self.patch_size_mixer(logi_patch_size1)
                patch_num_mx_list.append(patch_num_mx)
                patch_size_mx_list.append(patch_size_mx)


        patch_num_distribution_list = list(_flatten(patch_num_distribution_list))
        patch_size_distribution_list = list(_flatten(patch_size_distribution_list))
        patch_num_mx_list = list(_flatten(patch_num_mx_list))
        patch_size_mx_list = list(_flatten(patch_size_mx_list))

        if self.output_attention:
            return (
                patch_num_distribution_list,
                patch_size_distribution_list,
                patch_num_mx_list,
                patch_size_mx_list,
            )
        else:
            return None
