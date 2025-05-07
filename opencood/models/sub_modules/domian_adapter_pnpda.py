import torch

from opencood.models.fuse_modules.fusion_in_one import warp_affine_simple
from opencood.models.fuse_modules.pyramid_fuse import regroup
import math
from einops import einsum, rearrange
# import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.resize_net import ResizeNet
from opencood.utils.pe import PositionEmbeddingLearned, PositionEmbeddingSine
from opencood.models.sub_modules.base_transformer import FeedForward, PreNormResidual

class CrossAttention(nn.Module):
    def __init__(self, d_model, heads, d_ff, qkv_bias):
        super().__init__()

        dim_head = d_ff // heads
        self.scale = dim_head**-0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_ff, bias=qkv_bias)
        )
        self.to_k = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_ff, bias=qkv_bias)
        )
        self.to_v = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_ff, bias=qkv_bias)
        )

        self.proj = nn.Linear(d_ff, d_model)

    def forward(self, q, k, v, skip=None):
        """
        q: (b X Y W1 W2 d)
        k: (b x y w1 w2 d)
        v: (b x y w1 w2 d)
        return: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        _, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width
        # print(q.shape, k.shape)

        # flattening
        q = rearrange(q, "b x y w1 w2 d -> b (x y) (w1 w2) d")
        k = rearrange(k, "b x y w1 w2 d -> b (x y) (w1 w2) d")
        v = rearrange(v, "b x y w1 w2 d -> b (x y) (w1 w2) d")

        # Project with multiple heads
        q = self.to_q(q)  # b (X Y) (W1 W2) (heads dim_head)
        k = self.to_k(k)  # b (X Y) (w1 w2) (heads dim_head)
        v = self.to_v(v)  # b (X Y) (w1 w2) (heads dim_head)
        # print(q.shape,k.shape,v.shape)

        # Group the head dim with batch dim
        q = rearrange(q, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        k = rearrange(k, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        v = rearrange(v, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        # print(q.shape,k.shape,v.shape)

        # cross attention between cav and ego feature
        dot = self.scale * torch.einsum(
            "b l Q d, b l K d -> b l Q K", q, k
        )  # b (X Y) (W1 W2) (w1 w2)
        att = dot.softmax(dim=-1)

        a = torch.einsum("b n Q K, b n K d -> b n Q d", att, v)  # b (X Y) (W1 W2) d
        # print('a',a.shape)
        a = rearrange(a, "(b m) ... d -> b ... (m d)", m=self.heads, d=self.dim_head)
        # print('a',a.shape)
        a = rearrange(
            a,
            " b (x y) (w1 w2) d -> b x y w1 w2 d",
            x=q_height,
            y=q_width,
            w1=q_win_height,
            w2=q_win_width,
        )
        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + skip

        return z


class SelfAttention(nn.Module):
    def __init__(self, d_model, heads, d_ff, drop_out):
        super().__init__()

        dim_head = d_ff // heads
        self.scale = dim_head**-0.5

        self.heads = heads
        self.dim_head = dim_head
        
        self.to_q = nn.Linear(d_model, d_ff)
        self.to_k = nn.Linear(d_model, d_ff)
        self.to_v = nn.Linear(d_model, d_ff)
        
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(drop_out))
        self.to_out = nn.Sequential(
            nn.Linear(d_ff, d_model, bias=False), nn.Dropout(drop_out)
        )


    def forward(self, q,k,v):
        """
        q: (b h w d)
        k: (b h w d)
        v: (b h w d)
        return: (b h w d)
        """
        assert k.shape == v.shape
        _, q_height, q_width, _ = q.shape
        _, kv_height, kv_width, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        
        # Project with multiple heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # Group the head dim with batch dim
        q = rearrange(q, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        k = rearrange(k, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
        v = rearrange(v, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)

        # cross attention between cav and ego feature
        dot = self.scale * torch.einsum("b l Q d, b l K d -> b l Q K", q, k)
        att = self.attend(dot)

        a = torch.einsum("b n Q K, b n K d -> b n Q d", att, v)  # (b*head h w d/head)
        a = rearrange(a, "(b m) ... d -> b ... (m d)", m=self.heads, d=self.dim_head)
        # print('a',a.shape)

        return self.to_out(a)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.win_size = 2

        self.prenorm1 = nn.LayerNorm(d_model)
        self.prenorm2 = nn.LayerNorm(d_model)
        self.mlp_1 = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )

        self.cross_win_1 = CrossAttention(d_model, num_heads, d_ff, qkv_bias=True)
        self.cross_win_2 = CrossAttention(d_model, num_heads, d_ff, qkv_bias=True)

        self.win_size = 2

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, ego, cav_feature, pos=None):
        """
        Parameters
        ----------
        ego : b h w c
        cav_feature : b h w c
        """
        query = ego
        key = cav_feature
        value = cav_feature
        # print('query',query.shape)
        # print('key',key.shape)

        # local attention
        query = rearrange(
            query,
            "b (x w1) (y w2) d  -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition
        key = rearrange(
            key,
            "b (x w1) (y w2) d  -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition
        value = rearrange(
            value,
            "b (x w1) (y w2) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition
        pos = rearrange(
            pos,
            "b (x w1) (y w2) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition

        key = rearrange(
            self.cross_win_1(
                self.with_pos_embed(query, pos),
                self.with_pos_embed(key, pos),
                value,
                skip=query,
            ),
            "b x y w1 w2 d  -> b (x w1) (y w2) d",
        )
        key = self.prenorm1(key + self.mlp_1(key))
        query = rearrange(query, "b x y w1 w2 d  -> b (x w1) (y w2) d")
        value = rearrange(value, "b x y w1 w2 d  -> b (x w1) (y w2) d")
        pos = rearrange(pos, "b x y w1 w2 d  -> b (x w1) (y w2) d")

        # global attention
        query = rearrange(
            query,
            "b (w1 x) (w2 y) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )
        key = rearrange(
            key,
            "b (w1 x) (w2 y) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )
        value = rearrange(
            value,
            "b (w1 x) (w2 y) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )
        pos = rearrange(
            pos,
            "b (w1 x) (w2 y) d -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )
        key = rearrange(
            self.cross_win_2(
                self.with_pos_embed(query, pos),
                self.with_pos_embed(key, pos),
                value,
                skip=query,
            ),
            "b x y w1 w2 d  -> b (w1 x) (w2 y) d",
        )
        key = self.prenorm2(key + self.mlp_2(key))

        key = rearrange(key, "b h w d -> b d h w")

        return key



class TransformerEncoder(nn.Module):
    def __init__(self, args) -> None:
        super(TransformerEncoder, self).__init__()
        self.d_model = args["d_model"]
        self.num_heads = args["num_heads"]
        self.d_ff = args["d_ff"]
        self.num_layers = args["num_layers"]
        self.dropout = args["dropout"]
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.pos_embed_sin = PositionEmbeddingSine(d_model=self.d_model)

    def forward(self, x):
        # b,c,h,w
        x = x.permute(0, 2, 3, 1).contiguous()
        pos = self.pos_embed_sin(x)
        # print(x.shape,pos.shape)

        for i, layer in enumerate(self.layers):
            x = layer(x, pos)

        return x.permute(0, 3, 1, 2)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attention = SelfAttention(d_model, num_heads, d_ff, dropout)
        self.feedforward = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pos):
        q = k = self.with_pos_embed(x, pos)
        _x = self.self_attention(q, k, x)
        x = self.norm1(_x + x)
        _x = self.feedforward(x)
        x = self.norm2(x + _x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, args) -> None:
        super(TransformerDecoder, self).__init__()
        self.d_model = args["d_model"]
        self.num_heads = args["num_heads"]
        self.d_ff = args["d_ff"]
        self.num_layers = args["num_layers"]
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.pos_embed_sin = PositionEmbeddingSine(d_model=self.d_model)

    def forward(self, ego_feature, cav_feature):
        # b,h,w,c
        ego_feature = ego_feature.permute(0, 2, 3, 1).contiguous()
        cav_feature = cav_feature.permute(0, 2, 3, 1).contiguous()
        pos = self.pos_embed_sin(ego_feature)
        output = cav_feature

        for i, layer in enumerate(self.layers):
            output = layer(ego_feature, output, pos)

        return output


class DomianAdapter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.local_modality = args['local_modality']
        self.momentum_ratio = args['momentum_ratio']
        
        share_feat_dim = args['share_feat_dim']
        self.local_dim = share_feat_dim[self.local_modality]

        modality_name_list = ['m1', 'm2', 'm3', 'm4']
        neb_modality_name_list = list(set(modality_name_list) - set(self.local_modality))
        self.neb_modality_name_list = neb_modality_name_list
        for modality_name in neb_modality_name_list:
            setattr(self, f'resizer_for_{modality_name}', ResizeNet(share_feat_dim[modality_name], self.local_dim, args['resizer']['unify_method']))
            
            args['encoder']['d_model'] = self.local_dim
            args['decoder']['d_model'] = self.local_dim
            setattr(self, f'enhancer_for_{modality_name}', TransformerEncoder(args['encoder']))
            setattr(self, f'converter_for_{modality_name}', TransformerEncoder(args['encoder']))
            setattr(self, f'calibrator_for_{modality_name}', TransformerDecoder(args['decoder']))

            # enchancer参数与converter参数一致, 由enhancer参数更新converter参数
            # for para_enhancer, para_converter in zip(eval(f'self.enhancer_for_{modality_name}.parameters()'), \
            #                                          eval(f'self.converter_for_{modality_name}.parameters()')):
            #     para_enhancer.data.copy_(para_converter.data)

    def _momentum_update_enhancer(self):
        """
        Momentum update of the enhancer projection
        """
        for modality_name in self.neb_modality_name_list:
            for para_enhancer, para_converter in zip(eval(f'self.enhancer_for_{modality_name}.parameters()'), \
                                                     eval(f'self.converter_for_{modality_name}.parameters()')):
                para_enhancer.data.copy_(para_enhancer.data * self.momentum_ratio + para_converter.data * (
                1.0 - self.momentum_ratio ))


    def forward(self, x, agent_modality_list, record_len, affine_matrix):
        """Use param of converter to update param of enhancer"""
        self._momentum_update_enhancer()

        """Ego feat enhance, neb feat convert"""
        _, C, H, W = x.shape
        B, L = affine_matrix.shape[:2]
        split_x = regroup(x, record_len)
        # score = torch.sum(score, dim=1, keepdim=True)
        # split_score = regroup(score, record_len)
        # batch_node_features = split_x
        mode_idx = 0
        out = []
        query_for_train = []
        key_for_train = []
        adapt_agent_idx = [] # 与ego模态不同的neb的索引
        qk_pad = torch.ones_like(split_x[0][0].unsqueeze(0)) # 填充特征, 使之与label中pos_region_ranges的batch对齐, 且不影响训练
        qk_pad.requires_grad = False
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            batch_features = split_x[b]
            batch_out = []

            # 提取ego
            ego = batch_features[0].unsqueeze(0) # 1 h  w
            ego =  eval(f'self.enhancer_for_m2')(ego)
            batch_out.append(ego)
            mode_idx = mode_idx + 1

            batch_query = [qk_pad] # 填充特征, 使之与label中pos_region_ranges的batch对齐, 且不影响训练
            batch_key = [qk_pad]
            # ego = self.enhancer(ego) 

            if N > 1:
                # 提取neb, 若模态与ego不同, 转换
                # score = split_score[b]
                t_matrix = affine_matrix[b][:N, :N, :, :]
                # ego 映射到neb视角, 作为查询
                i = 0 # ego
                ego_expand = ego.expand(N-1, -1, -1, -1)
                ego_in_neb = warp_affine_simple(ego_expand,
                                                t_matrix[1:, i, :, :],
                                                (H, W))
                

                for cav_id in range(0, N-1):
                    cav_mode = agent_modality_list[mode_idx]
                    if cav_mode != self.local_modality:
                        key = eval(f'self.resizer_for_{cav_mode}')\
                            (batch_features[cav_id + 1].unsqueeze(0), (C, H, W))
                        key = eval(f'self.converter_for_{cav_mode}')(key)
                        
                        """这里enhancer的用法与pnpda原文处理不同
                        输入FuseNet的特征是enhancer之前的ego而不是之后的,
                        因为一方面, 在协作方智能体有多个类型时, enhancer需要重新适配, 就影响了之前模态的表现
                        另一方面, 直接输入原始的ego特征更有利于FusionNet和Decoder识别"""
                        batch_out.append(key)
                        
                        query = eval(f'self.enhancer_for_{cav_mode}')\
                            (ego_in_neb[cav_id].unsqueeze(0))
                        key = eval(f'self.calibrator_for_{cav_mode}')(query, key)

                        batch_key.append(key)
                        batch_query.append(query)
                        adapt_agent_idx.append(mode_idx)

                    elif cav_mode == self.local_modality:
                        key =  eval(f'self.enhancer_for_m2')(batch_features[cav_id + 1].unsqueeze(0))
                        batch_out.append(key)

                        # batch_out.append(batch_features[cav_id + 1].unsqueeze(0))
                        batch_key.append(qk_pad)
                        batch_query.append(qk_pad)


                    mode_idx = mode_idx + 1

            # neb = batch_features[1:]
            # neb = self.converter(neb)

            batch_out = torch.cat(batch_out, dim=0)
            out.append(batch_out)

            batch_query = torch.cat(batch_query, dim=0)
            batch_key = torch.cat(batch_key, dim=0)
            query_for_train.append(batch_query)
            key_for_train.append(batch_key)

        out = torch.cat(out, dim=0)
        query_for_train = torch.cat(query_for_train, dim=0)
        key_for_train = torch.cat(key_for_train, dim=0)

        return out, query_for_train, key_for_train, adapt_agent_idx
    
if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = {
        "momentum_ratio": 0.8,
        "encoder":
          {"num_layers": 2,
          "num_heads": 8,
          "d_ff": 128,
          "d_model": 64,
          "dropout": 0.1},
        "decoder":
          {"num_layers": 1,
          "num_heads": 8,
          "d_model": 64,
          "d_ff": 128}
    }
    model = DomianAdapter(args)
