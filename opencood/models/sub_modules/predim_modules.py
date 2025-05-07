"""
This class is about swap fusion applications
"""
import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange
import torch.nn.functional as F

from opencood.models.fuse_modules.wg_fusion_modules import sc_padding, sc_unpadding
from opencood.tools.feature_show import feature_show
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone



class CrossAttentionPerdim(nn.Module):
    def __init__(
        self, dim, heads, dim_head, qkv_bias=False, rel_pos_emb=False, norm=nn.LayerNorm
    ):
        super().__init__()

        self.scale = dim_head**-0.5

        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb

        self.to_q = nn.Sequential(
            norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias)
        )
        self.to_k = nn.Sequential(
            norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias)
        )
        self.to_v = nn.Sequential(
            norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias)
        )

        self.proj = nn.Linear(heads * dim_head, dim)

    def add_rel_pos_emb(self, x):
        return x

    def forward(self, q, k, v, skip=None):
        """
        q: (b k X Y W1 W2 d)
        k: (b k x y w1 w2 d)
        v: (b k x y w1 w2 d)
        return: (b k X Y W1 W2 d)
        """
        assert k.shape == v.shape
        _, keyfeat_dim, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _,_, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        # flattening
        q = rearrange(q, "b k x y w1 w2 d -> b k (x y) (w1 w2) d")
        k = rearrange(k, "b k x y w1 w2 d -> b k (x y) (w1 w2) d")
        v = rearrange(v, "b k x y w1 w2 d -> b k (x y) (w1 w2) d")

        # Project with multiple heads
        q = self.to_q(q)  # b k (X Y) (W1 W2) (heads dim_head)
        k = self.to_k(k)  # b k (X Y) (w1 w2) (heads dim_head)
        v = self.to_v(v)  # b k (X Y) (w1 w2) (heads dim_head)
        # print(q.shape,k.shape,v.shape)

        # Group the head dim with batch dim
        q = rearrange(q, "b k ... (m d) -> (b k m) ... d", m=self.heads, d=self.dim_head)
        k = rearrange(k, "b k ... (m d) -> (b k m) ... d", m=self.heads, d=self.dim_head)
        v = rearrange(v, "b k ... (m d) -> (b k m) ... d", m=self.heads, d=self.dim_head)
        # print(q.shape,k.shape,v.shape)

        # cross attention between cav and ego feature
        dot = self.scale * torch.einsum(
            "b l Q d, b l K d -> b l Q K", q, k
        )  # b k (X Y) (W1 W2) (w1 w2)

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)

        a = torch.einsum("b n Q K, b n K d -> b n Q d", att, v)  # b k (X Y) (W1 W2) d
        # print('a',a.shape)
        a = rearrange(a, "(b k m) ... d -> b k ... (m d)", k=keyfeat_dim, \
            m=self.heads, d=self.dim_head)
        # print('a',a.shape)
        a = rearrange(
            a,
            " b k (x y) (w1 w2) d -> b k x y w1 w2 d",
            x=q_height,
            y=q_width,
            w1=q_win_height,
            w2=q_win_width,
        )
        # print('a',a.shape)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + skip
        return z


class CrossAttenPerdimBlock(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention.
    """

    def __init__(self, dim, dim_heads, heads, qkv_bias, win_size):
        super(CrossAttenPerdimBlock, self).__init__()
        self.win_size = win_size

        self.prenorm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )


        self.cross_win = CrossAttentionPerdim(dim, heads, dim_heads, qkv_bias)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, ego, cav_feature):
        """
        Parameters
        ----------
        ego : b * k * d * h * w
        cav_feature : b * k * d * h *w
        """
        query = cav_feature
        key = ego
        value = ego

        # local attention
        query = rearrange(
            query,
            "b k d (x w1) (y w2) -> b k x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition
        key = rearrange(
            key,
            "b k d (x w1) (y w2) -> b k x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition
        value = rearrange(
            value,
            "b k d (x w1) (y w2) -> b k x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition

        query = rearrange(
            self.cross_win(query, key, value, skip=query),
            "b k x y w1 w2 d  -> b k (x w1) (y w2) d",
        )
        query = self.prenorm(query)
        
        query = query + self.ff(query)
        query = self.post_norm(query)

        query = rearrange(query, "b k h w d -> b k d h w")
        
        return query
        


class KeyfeatAlignPerdim(nn.Module):
    def __init__(self, args):
        super(KeyfeatAlignPerdim, self).__init__()

        
        self.depth = args["num_of_blocks"]

        # block related
        input_dim = args['expand_dim']        
        heads = args['heads']
        dim_head = input_dim // heads
        window_size = args["window_size"]
        self.window_size = window_size
        
        self.layers = nn.ModuleList([])
        for i in range(self.depth):
            self.layers.append(
                CrossAttenPerdimBlock(
                    input_dim, dim_head, heads, True, window_size
                )
            )


        # dim ascending
        expand_dim = args['expand_dim']
        self.perdim_ascend_ego = nn.Linear(1, expand_dim)
        self.perdim_ascend_cav = nn.Linear(1, expand_dim)
        

        # out: 对特征最终维进行调整, 并降维
        """b k e h w -> b h w k e -> b h w k e -> b h w k 1"""
        self.out = nn.Sequential(
            Rearrange("b k e h w -> b h w k e"),
            nn.LayerNorm(expand_dim),
            nn.Linear(expand_dim, expand_dim),
            nn.Linear(expand_dim, 1)
        )

    def forward(self, ego_feature, cav_feature):
        # 调整特征大小, 使可以按窗格裁剪
        _, padding_pos = sc_padding(cav_feature, self.window_size)
        cav_feature = F.pad(cav_feature, padding_pos)
        ego_feature = F.pad(ego_feature, padding_pos)
        
        # 每个维度分别执行converter体现在了额外增加的维度k中
        # 分别升维keyfeat的每个维度, 并将升维后的维度e作为特征, 原有的特征的k维作为batch之一
        """b k h w -> b h w k 1 -> b h w k e -> b k e h w"""
        ego_feature = rearrange(ego_feature, 'b k h w -> b h w k')
        ego_feature = ego_feature.unsqueeze(-1)
        ego_feature = self.perdim_ascend_ego(ego_feature)
        ego_feature = rearrange(ego_feature, 'b h w k e -> b k e h w')
        
        cav_feature = rearrange(cav_feature, 'b k h w -> b h w k')
        cav_feature = cav_feature.unsqueeze(-1)
        cav_feature = self.perdim_ascend_cav(cav_feature)
        cav_feature = rearrange(cav_feature, 'b h w k e -> b k e h w')
        
        
        # 对每个维度, 分别在高维空间计算跨模态注意力
        x = cav_feature
        for block in self.layers:
            x = block(ego_feature, x)
        
        
        # 输出处理, 在每个维度的高维表示分别执行注意力, 再降维
        x = sc_unpadding(x, padding_pos)
        x = self.out(x)
        
        """b h w k 1 -> b h w k -> b k h w"""
        x = x.squeeze(-1)
        x = rearrange(x, 'b h w k -> b k h w')
          
        return x


class MlpPerdim(nn.Module):
    def __init__(self, expand_dim) -> None:
        super().__init__()
        """
        对特征的每个维度分别进行Mlp
        实现方法: 
        将单层特征分别映射到更高维度, 在高维度进行mlp交互, 最后再映射回一维, 依次捕获更丰富的交互关系
        """
                
        """b h w c 1 -> b h w c e -> b h w c 1"""
        self.ascend_interact = nn.Sequential(
            nn.Linear(1, expand_dim),
            nn.Linear(expand_dim, expand_dim),
            nn.Linear(expand_dim, 1)
        )       
        
        
    def forward(self, x):
        """
        Input: x -> b c h w
        """

        """ b c h w -> b h w c 1 """
        x = rearrange(x, 'b c h w -> b h w c')
        x = x.unsqueeze(-1)
        
        
        x = self.ascend_interact(x)
        
        """ b h w c 1 -> b c h w """
        x = x.squeeze(-1)
        x = rearrange(x, 'b h w c -> b c h w')
        
        return x


class Gate2PubSemantic(nn.Module):
    def __init__(self, args) -> None:
        super(Gate2PubSemantic, self).__init__()
        
        """select and modify key feat to pub space in each dim"""
        
        # input_dim = args['input_dim']
        gatemode = args['gatemode']
        expand_dim = args['expand_dim']
        
        self.proj = MlpPerdim(expand_dim)
        
        if gatemode == 'silu':
            self.gate = nn.SiLU()
    
    def forward(self, x):
        """
        x: b c h w 
        return: gate score
        """
        
        x = self.proj(x)
        score = self.gate(x)
        
        return score



# if __name__ == "__main__":
#     import os

#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     ego = torch.rand(2, 8, 100, 352)  # .cuda()
#     cav = torch.rand(2, 8, 100, 352)  # .cuda()


    # h = 50
    # w = 176
    # # local attention
    # query = rearrange(
    #     ego, "b d (x w1) (y w2) -> b x y w1 w2 d", x=h, w2=w
    # )  # window partition
    # key = rearrange(
    #     cav, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=h, w2=w
    # )  # window partition
    # print(query.shape)
    # print(key.shape)

    # args = {
    #     "input_dim": 256,
    #     "window_size": 8,
    #     "dim_head": 32,
    #     "heads": 16,
    #     "depth": 1,
    # }
    
    # args = {
    #     'num_of_blocks': 1,
    #     'expand_dim': 4,
    #     'window_size': 2,
    #     'heads': 1,
    #     'drop_out': 0.2
    # }
    
    # model = KeyfeatAlignPerdim(args)
    # output = model(ego, cav)
    # print(output)
