from extractor import BasicEncoder
from position_encoding import build_position_encoding

import argparse
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import copy
from typing import Optional


class attnLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_list = nn.ModuleList([copy.deepcopy(nn.MultiheadAttention(d_model, nhead, dropout=dropout)) for i in range(2)])
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2_list = nn.ModuleList([copy.deepcopy(nn.LayerNorm(d_model)) for i in range(2)])

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2_list = nn.ModuleList([copy.deepcopy(nn.Dropout(dropout)) for i in range(2)])
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_list, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, memory_pos=None):
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        for memory, multihead_attn, norm2, dropout2, m_pos in zip(memory_list, self.multihead_attn_list, self.norm2_list, self.dropout2_list, memory_pos):
            tgt2 = multihead_attn(query=self.with_pos_embed(tgt, pos),
                                       key=self.with_pos_embed(memory, m_pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + dropout2(tgt2)
            tgt = norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, memory_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory_list, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, memory_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory_list, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, memory_pos)
        return self.forward_post(tgt, memory_list, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, memory_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransDecoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransDecoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf, query_embed):
        pos = self.position_embedding(torch.ones(imgf.shape[0], imgf.shape[2], imgf.shape[3]).bool().cuda())  # torch.Size([1, 128, 36, 36])
        
        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            query_embed = layer(query_embed, [imgf], pos=pos, memory_pos=[pos, pos])
        query_embed = query_embed.permute(1, 2, 0).reshape(bs, c, h, w)

        return query_embed


class TransEncoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransEncoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, imgf):
        pos = self.position_embedding(torch.ones(imgf.shape[0], imgf.shape[2], imgf.shape[3]).bool().cuda())  # torch.Size([1, 128, 36, 36])
        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)  
        pos = pos.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            imgf = layer(imgf, [imgf], pos=pos, memory_pos=[pos, pos])
        imgf = imgf.permute(1, 2, 0).reshape(bs, c, h, w)

        return imgf


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, imgf, coords1):
        mask = .25 * self.mask(imgf)  # scale mask to balence gradients
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow

        return mask, coords1


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W
        

class GeoTr(nn.Module):
    def __init__(self):
        super(GeoTr, self).__init__()

        self.hidden_dim = hdim = 256

        self.fnet = BasicEncoder(output_dim=hdim, norm_fn='instance')

        self.encoder_block = ['encoder_block' + str(i) for i in range(3)]
        for i in self.encoder_block:
            self.__setattr__(i, TransEncoder(2, hidden_dim=hdim))  
        self.down_layer = ['down_layer' + str(i) for i in range(2)]
        for i in self.down_layer:
            self.__setattr__(i, nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1))
           
        self.decoder_block = ['decoder_block' + str(i) for i in range(3)]
        for i in self.decoder_block:
            self.__setattr__(i, TransDecoder(2, hidden_dim=hdim))  
        self.up_layer = ['up_layer' + str(i) for i in range(2)]
        for i in self.up_layer:
            self.__setattr__(i, nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
           
        self.query_embed = nn.Embedding(81, self.hidden_dim)
        
        self.update_block = UpdateBlock(self.hidden_dim)
                                    
    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2) 

        up_flow = F.unfold(8 * flow, [3, 3], padding=1) 
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W) 

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1):
        fmap = self.fnet(image1)
        fmap = torch.relu(fmap)
        
        # fmap = self.TransEncoder(fmap)
        fmap1 = self.__getattr__(self.encoder_block[0])(fmap)
        fmap1d = self.__getattr__(self.down_layer[0])(fmap1)
        fmap2 = self.__getattr__(self.encoder_block[1])(fmap1d)
        fmap2d = self.__getattr__(self.down_layer[1])(fmap2)
        fmap3 = self.__getattr__(self.encoder_block[2])(fmap2d)
        
        query_embed0 = self.query_embed.weight.unsqueeze(1).repeat(1, fmap3.size(0), 1)
        fmap3d_ = self.__getattr__(self.decoder_block[0])(fmap3, query_embed0)
        fmap3du_ = self.__getattr__(self.up_layer[0])(fmap3d_).flatten(2).permute(2, 0, 1) 
        fmap2d_ = self.__getattr__(self.decoder_block[1])(fmap2, fmap3du_)
        fmap2du_ = self.__getattr__(self.up_layer[1])(fmap2d_).flatten(2).permute(2, 0, 1)         
        fmap_out = self.__getattr__(self.decoder_block[2])(fmap1, fmap2du_)   
        
        # convex upsample baesd on fmap_out
        coodslar, coords0, coords1 = self.initialize_flow(image1)
        coords1 = coords1.detach()
        mask, coords1 = self.update_block(fmap_out, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up

        return bm_up 
