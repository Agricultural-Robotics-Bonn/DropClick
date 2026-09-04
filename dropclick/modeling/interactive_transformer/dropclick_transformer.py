# Adapted by Amit Rana from: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
# Adapted by Patrick Zimmer from https://github.com/amitrana001/DynaMITe

import logging
import os
import time as timer
import random
import einops
import numpy as np
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch

from detectron2.utils.registry import Registry
import detectron2.utils.comm as comm
from torch import nn, Tensor
from torch.nn import functional as F
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.config import configurable
from detectron2.layers import Conv2d
from einops import repeat
import copy

from .position_encoding import PositionEmbeddingSine
from .descriptor_initializer import AvgClicksPoolingInitializer
from dropclick.utils.train_utils import get_pos_tensor_coords, get_spatiotemporal_embeddings
from .utils import INTERACTIVE_TRANSFORMER_REGISTRY, MLP
from .encoder import Encoder
from .decoder import Decoder

@INTERACTIVE_TRANSFORMER_REGISTRY.register()
class DropClickTransformer(nn.Module):

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        use_decoder, 
        dec_layers,
        dec_scale_factor,
        use_static_bg_queries: bool,
        num_static_bg_queries: int,
        num_static_fg_queries: int,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        enc_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        positional_embeddings: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            use_decoder: whether to use decoder
            dec_layers: number of decoder layers
            dec_scale_factor: scaling factor for mask_features before using in decoder
            use_static_bg_queries: whether to use learnt background queries
            num_static_bg_queries: number of learnt background queries
            hidden_dim: Transformer feature dimension
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            positional_embeddings: type of positonal embeddings for clicks coordinates 
        """
        super().__init__()

        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        self.positional_embeddings = positional_embeddings

        # iterative
        self.num_static_bg_queries = num_static_bg_queries
        self.num_static_fg_queries = num_static_fg_queries
        
        # Reverse Cross Attn
        self.use_decoder = use_decoder
        self.dec_layers = dec_layers
        self.dec_scale_factor = dec_scale_factor

        self.num_heads = nheads
        self.enc_layers = enc_layers
        self.encoder = Encoder(hidden_dim, dim_feedforward, nheads, self.enc_layers, pre_norm)
        if self.use_decoder:
            self.decoder = Decoder(hidden_dim, nheads, self.dec_layers, pre_norm)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.query_descriptors_initializer = AvgClicksPoolingInitializer()
        
        self.queries_nonlinear_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if self.positional_embeddings == "spatio_temporal":
            self.ca_qpos_sine_proj = nn.Linear(3*(hidden_dim//2), hidden_dim)
        elif self.positional_embeddings in ["temporal","spatial"]:
            self.ca_qpos_sine_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # learnable query p.e.
        self.register_parameter("query_embed", nn.Parameter(torch.zeros(hidden_dim), True))
       
        self.use_static_bg_queries = use_static_bg_queries
        if self.use_static_bg_queries:
            self.register_parameter("static_bg_pe", nn.Parameter(torch.zeros(self.num_static_bg_queries, hidden_dim), True))
            self.register_parameter("static_bg_query", nn.Parameter(torch.zeros(self.num_static_bg_queries,hidden_dim), True))
        self.register_parameter("bg_query", nn.Parameter(torch.zeros(hidden_dim), False))

        ## learnable fg queries
        self.num_static_fg_queries = num_static_fg_queries
        self.register_parameter("static_fg_pe", nn.Parameter(torch.zeros(self.num_static_fg_queries, hidden_dim), True))
        self.register_parameter("static_fg_query", nn.Parameter(torch.zeros(self.num_static_fg_queries, hidden_dim), True))
        

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.query_embed)
        if self.use_static_bg_queries:
            nn.init.normal_(self.static_bg_pe)
            nn.init.xavier_uniform_(self.static_bg_query)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["num_classes"] = 1  # always 1 for us
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # DECODER
        ret["use_decoder"] =  cfg.MODEL.MASK_FORMER.DECODER.USE_DECODER
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DECODER.DEC_LAYERS
        ret["dec_scale_factor"] = cfg.MODEL.MASK_FORMER.DECODER.DEC_SCALE_FACTOR

        # Iterative Pipeline
        ret["positional_embeddings"] = cfg.ITERATIVE.TRAIN.POSITIONAL_EMBED

        ret["use_static_bg_queries"] = cfg.ITERATIVE.TRAIN.USE_STATIC_BG_QUERIES
        ret["num_static_bg_queries"] = cfg.ITERATIVE.TRAIN.NUM_STATIC_BG_QUERIES
        ret["num_static_fg_queries"] = cfg.ITERATIVE.TRAIN.NUM_STATIC_FG_QUERIES
        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.ENC_LAYERS >= 1
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, data, images, num_instances, x, mask_features, num_clicks_per_object=None,
                 fg_coords = None, max_timestamp=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        
        src = []
        pos = []
        size_list = []

        ### drop some clicks
        if data[0]['instances'].has('gt_click_missing'):
            fg_coords_filtered = []
            for i,img in enumerate(data):
                fg_coords_filtered_ = [ fg_coords[i][j] for j, is_missing in enumerate(img['instances'].gt_click_missing) if not is_missing ]
                fg_coords_filtered.append( fg_coords_filtered_ )
            fg_coords = fg_coords_filtered

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        if self.training:                       
            outputs = self.iterative_batch_forward(x, src, pos, size_list, mask_features, fg_coords, 
                                                    max_timestamp
                    )
        else:
            outputs = self.iterative_batch_forward(x, src, pos, size_list, mask_features, fg_coords,
                                                   max_timestamp)
        return outputs, num_clicks_per_object

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.layer_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    def iterative_batch_forward(self, x, src, pos, size_list, mask_features, fg_coords=None, max_timestamp=None):

        _, bs, _ = src[0].shape
        B, C, H, W = mask_features.shape
        height = 4*H
        width = 4*W
        
        descriptors = self.query_descriptors_initializer(x, fg_coords)
        max_queries_batch = max([desc.shape[1] for desc in descriptors])
        for i, desc in enumerate(descriptors):
            if self.use_static_bg_queries:
                bg_queries = repeat(self.bg_query, "C -> 1 L C", L=max_queries_batch-desc.shape[1])
            else:
                bg_queries = repeat(self.bg_query, "C -> 1 L C", L=max_queries_batch+1-desc.shape[1])
            descriptors[i] = torch.cat((descriptors[i], bg_queries), dim=1)
        output = torch.cat(descriptors, dim=0)
        
        query_embed = repeat(self.query_embed, "C -> Q N C", N=bs, Q=output.shape[1])
        
        if self.positional_embeddings:
            normalized_click_coords = get_pos_tensor_coords(fg_coords, output.shape[1], height, width, output.device, max_timestamp=max_timestamp) # bsxQx3
            if output.shape[1]:  ## if descs are not None
                pos_coord_embed = get_spatiotemporal_embeddings(normalized_click_coords.permute(1,0,2), self.positional_embeddings) # Q x bs x C
                pos_coord_embed = self.ca_qpos_sine_proj(pos_coord_embed.to(query_embed.dtype))
            else:
                pos_coord_embed = self.ca_qpos_sine_proj(torch.empty((0,B,self.ca_qpos_sine_proj.in_features), dtype=query_embed.dtype, device=query_embed.device) )  ## make sure those layers are being used even when no clicks are available. otherwise leads to ddp errors
            
            query_embed = query_embed + pos_coord_embed
                
        if self.use_static_bg_queries:
            static_bg_pe = repeat(self.static_bg_pe, "Bg C -> Bg N C", N=bs)
            query_embed = torch.cat((query_embed,static_bg_pe),dim=0)
            static_bg_queries = repeat(self.static_bg_query, "Bg C -> N Bg C", N=bs)
            output = torch.cat((output,static_bg_queries), dim=1)

        ## fg queries
        static_fg_pe = repeat(self.static_fg_pe, "Bg C -> Bg N C", N=bs)
        query_embed = torch.cat((static_fg_pe, query_embed),dim=0)
        static_fg_queries = repeat(self.static_fg_query, "Bg C -> N Bg C", N=bs)
        output = torch.cat((static_fg_queries, output), dim=1)

    
        # NxQxC -> QxNxC
        output = self.queries_nonlinear_projection(output).permute(1,0,2)
        # query positional embedding QxNxC
        
        predictions_mask = []
        predictions_class = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.enc_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.encoder.cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.encoder.self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.encoder.ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.enc_layers + 1

        if self.use_decoder:
            if self.dec_scale_factor > 1:
                scale_factor = self.dec_scale_factor
                mask_features = F.interpolate(mask_features, scale_factor=scale_factor, mode='bilinear', align_corners=False)
           
            mask_features = self.decoder((mask_features, output, query_embed))
            mask_features = einops.rearrange(mask_features,"(H W) B C -> B C H W", H=H, W=W, B=B).contiguous()
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
    
