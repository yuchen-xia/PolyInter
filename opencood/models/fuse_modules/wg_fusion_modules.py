"""
This class is about swap fusion applications
"""
from typing import Iterator
import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F

from opencood.models.sub_modules.base_transformer import FeedForward, PreNormResidual
from opencood.tools.feature_show import feature_show


class CrossAttention(nn.Module):
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
        q: (b X Y W1 W2 d)
        k: (b x y w1 w2 d)
        v: (b x y w1 w2 d)
        return: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        _, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

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

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
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
        # print('a',a.shape)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + skip
        return z


# swap attention -> max_vit
class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0.0, window_size=7):
        super().__init__()
        assert (
            dim % dim_head
        ) == 0, "dimension should be divisible by dimension per head"

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(
            grid, "j ... -> 1 j ..."
        )
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = (
            *x.shape,
            x.device,
            self.heads,
        )

        # flatten
        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, "b n (h d ) -> b h n d", h=h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, "i j h -> h i j")

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads
        out = rearrange(
            out, "b h (w1 w2) d -> b w1 w2 (h d)", w1=window_height, w2=window_width
        )

        # combine heads out
        out = self.to_out(out)

        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width)

class SwapFusionBlock(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention.
    """

    def __init__(self, input_dim, mlp_dim, dim_head, window_size, drop_out):
        super(SwapFusionBlock, self).__init__()
        # b = batch * max_cav
        self.block = nn.Sequential(
            # window attention, innner window
            Rearrange(
                "b d (x w1) (y w2) -> b x y w1 w2 d", w1=window_size, w2=window_size
            ),
            PreNormResidual(
                input_dim, Attention(input_dim, dim_head, drop_out, window_size)
            ),
            PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
            
            # grid attention, cross window
            Rearrange(
                "b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=window_size, w2=window_size
            ),
            PreNormResidual(
                input_dim, Attention(input_dim, dim_head, drop_out, window_size)
            ),
            PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
        )

    def forward(self, x, mask=None):
        x = self.block(x)
        return x

class CrossDomainSwapFusionBlock(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention.
    """

    def __init__(self, dim, dim_heads, heads, qkv_bias, win_size):
        super(CrossDomainSwapFusionBlock, self).__init__()
        self.win_size = win_size

        self.prenorm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )


        self.cross_win = CrossAttention(dim, heads, dim_heads, qkv_bias)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, ego, cav_feature):
        """
        Parameters
        ----------
        ego : b * c * h * w
        cav_feature : b * c* h *w
        """
        query = cav_feature
        key = ego
        value = ego

        # local attention
        query = rearrange(
            query,
            "b d (x w1) (y w2) -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition
        key = rearrange(
            key,
            "b d (x w1) (y w2) -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition
        value = rearrange(
            value,
            "b d (x w1) (y w2) -> b x y w1 w2 d",
            w1=self.win_size,
            w2=self.win_size,
        )  # window partition

        query = rearrange(
            self.cross_win(query, key, value, skip=query),
            "b x y w1 w2 d  -> b (x w1) (y w2) d",
        )
        query = self.prenorm(query)
        
        query = query + self.ff(query)
        query = self.post_norm(query)

        query = rearrange(query, "b h w d -> b d h w")
        
        return query


class CrossDomainFusionEncoder(nn.Module):
    def __init__(self, args):
        super(CrossDomainFusionEncoder, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args["depth"]

        # block related
        input_dim = args["input_dim"]
        heads = args["heads"]
        dim_head = args["dim_head"]
        window_size = args["window_size"]

        for i in range(self.depth):
            self.layers.append(
                CrossDomainSwapFusionBlock(
                    input_dim, dim_head, heads, True, window_size
                )
            )

        # mlp head
        self.mlp_head = nn.Sequential(
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange("b h w d -> b d h w"),
        )

    def forward(self, ego_feature, cav_feature):
        x = cav_feature
        for block in self.layers:
            x = block(ego_feature, x)
        return self.mlp_head(x)



def sc_padding(x, window_size):
    # calcute padding size and scaled h, w
    padding_left, padding_right, padding_top, padding_bottom = 0, 0, 0, 0
    _, _, h, w= x.size()
    h_sc, w_sc = h // window_size, w // window_size
    res_h = h % window_size
    
    if res_h > 0:
        h_sc = h_sc + 1
        padding_bottom = window_size - res_h
    res_w = w % window_size
    if res_w > 0:
        w_sc = w_sc + 1
        padding_right = window_size - res_w
    return [h_sc, w_sc], [padding_left, padding_right, padding_top, padding_bottom]

def sc_unpadding(x, padding):
    if len(x.shape) == 4:
        if padding[1] > 0:
            x = x[:, :, :, :-padding[1]]
        if padding[3] > 0:
            x = x[:, :, :-padding[3], :]
    
    if len(x.shape) == 5:
        if padding[1] > 0:
            x = x[:, :, :, :, :-padding[1]]
        if padding[3] > 0:
            x = x[:, :, :, :-padding[3], :]
    return x


class SwapFusionEncoder(nn.Module):
    """
    Data rearrange -> swap block -> mlp_head
    """

    def __init__(self, args):
        super(SwapFusionEncoder, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args["depth"]

        # block related
        input_dim = args["input_dim"]
        mlp_dim = args["mlp_dim"]
        window_size = args["window_size"]
        drop_out = args["drop_out"]
        dim_head = args["dim_head"]

        self.window_size = args["window_size"]
        self.mask = False

        for i in range(self.depth):
            block = SwapFusionBlock(input_dim, mlp_dim, dim_head, window_size, drop_out)
            self.layers.append(block)

        # mlp head
        self.mlp_head = nn.Sequential(
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange("b h w d -> b d h w"),
        )

    def forward(self, x, mask=None):
        # if cannot be divided by window_size, pad the feature
        _, padding_pos = sc_padding(x, self.window_size)
        x = F.pad(x, padding_pos)

        for stage in self.layers:
            x = stage(x, mask=mask)
        
        # unpad feature to origin size
        x = sc_unpadding(x, padding_pos)

        return self.mlp_head(x)


class VallianceCrossAttention(nn.Module):
    def __init__(
        self, dim, heads, dim_head, qkv_bias=False, rel_pos_emb=False, norm=nn.LayerNorm
    ):
        super().__init__()

        self.scale = dim_head**-0.5

        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb
        
    def add_rel_pos_emb(self, x):
        return x

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

        # flattening
        q = rearrange(q, "b x y w1 w2 d -> b (x y) (w1 w2) d")
        k = rearrange(k, "b x y w1 w2 d -> b (x y) (w1 w2) d")
        v = rearrange(v, "b x y w1 w2 d -> b (x y) (w1 w2) d")

        # Project with multiple heads
        # q = self.to_q(q)  # b (X Y) (W1 W2) (heads dim_head)
        # k = self.to_k(k)  # b (X Y) (w1 w2) (heads dim_head)
        # v = self.to_v(v)  # b (X Y) (w1 w2) (heads dim_head)
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

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)
        
        # feature_show(att[0], 'analysis/atten_fill/attmap')

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

        return a

class VallianceCrossDomainSwapFusionBlock(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention.
    """

    def __init__(self,
                 dim,
                 dim_heads,
                 heads,
                 qkv_bias,
                 win_size
                 ):
        super(VallianceCrossDomainSwapFusionBlock, self).__init__()
        self.win_size = win_size

        self.prenorm1 = nn.LayerNorm(dim)
        self.prenorm2 = nn.LayerNorm(dim)
        # self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim),
        #                            nn.GELU(), nn.Linear(2 * dim, dim))
        # self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim),
        #                            nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_1 = nn.Sequential(
                                   nn.GELU())
        self.mlp_2 = nn.Sequential(
                                   nn.GELU())

        self.cross_win_1 = VallianceCrossAttention(dim, heads, dim_heads, qkv_bias)
        self.cross_win_2 = VallianceCrossAttention(dim, heads, dim_heads, qkv_bias)
        self.feed_forward = FeedForward(dim, dim*2, 0.1)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, ego, cav_feature):
        """
        Parameters
        ----------
        ego : b * c * h * w
        cav_feature : b * c* h *w
        """
        query = cav_feature
        key = ego
        value = ego

        # local attention
        query = rearrange(query, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                          w1=self.win_size, w2=self.win_size)  # window partition
        key = rearrange(key, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                          w1=self.win_size, w2=self.win_size)  # window partition
        value = rearrange(value, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                          w1=self.win_size, w2=self.win_size)  # window partition

        query = rearrange(self.cross_win_1(query, key, value, skip=query),
                          'b x y w1 w2 d  -> b (x w1) (y w2) d')
        
        # q = rearrange(query, 'b h w d -> b d h w')
        # feature_show(q[0], 'analysis/atten_fill/query')
        
        query = query + self.mlp_1(self.prenorm1(query))
        
        
        # q = rearrange(query, 'b h w d -> b d h w')
        # feature_show(q[0], 'analysis/atten_fill/query_after_ff')
        
        key = rearrange(key, 'b x y w1 w2 d  -> b (x w1) (y w2) d')
        value = rearrange(value, 'b x y w1 w2 d  -> b (x w1) (y w2) d')

        # global attention
        query = rearrange(query, 'b  (w1 x) (w2 y) d -> b x y w1 w2 d',
                          w1=self.win_size, w2=self.win_size)
        key = rearrange(key, 'b  (w1 x) (w2 y) d -> b x y w1 w2 d',
                          w1=self.win_size, w2=self.win_size)
        value = rearrange(value, 'b  (w1 x) (w2 y) d -> b x y w1 w2 d',
                        w1=self.win_size, w2=self.win_size)
        query = rearrange(self.cross_win_2(query, key, value, skip=query),
                          'b x y w1 w2 d  -> b (w1 x) (w2 y) d')
        query = query + self.mlp_2(self.prenorm2(query))
        query = self.post_norm(query)



        query = rearrange(query, 'b h w d -> b d h w')
        return query


class VallianceFiller(nn.Module):
    def __init__(self, args):
        super(VallianceFiller, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args["depth"]

        # block related
        input_dim = args["input_dim"]
        heads = args["heads"]
        dim_head = args["dim_head"]
        window_size = args["window_size"]
        self.window_size = window_size

        for i in range(self.depth):
            self.layers.append(
                VallianceCrossDomainSwapFusionBlock(
                    input_dim, dim_head, heads, True, window_size
                )
            )


    def forward(self, ego_feature, cav_feature):
        _, padding_pos = sc_padding(cav_feature, self.window_size)
        cav_feature = F.pad(cav_feature, padding_pos)
        ego_feature = F.pad(ego_feature, padding_pos)
        
        x = cav_feature
        for block in self.layers:
            x = block(ego_feature, x)
            
        x = sc_unpadding(x, padding_pos)
        return x


class Filler(nn.Module):
    def __init__(self, args):
        super(Filler, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args["depth"]

        # block related
        input_dim = args["input_dim"]
        heads = args["heads"]
        dim_head = args["dim_head"]
        window_size = args["window_size"]
        self.window_size = window_size

        for i in range(self.depth):
            self.layers.append(
                CrossDomainSwapFusionBlock(
                    input_dim, dim_head, heads, True, window_size
                )
            )

        # mlp head
        self.mlp_head = nn.Sequential(
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 8 * input_dim),
            nn.Linear(8 * input_dim, input_dim),
            Rearrange("b h w d -> b d h w"),
        )

    def forward(self, ego_feature, cav_feature):
        _, padding_pos = sc_padding(cav_feature, self.window_size)
        cav_feature = F.pad(cav_feature, padding_pos)
        ego_feature = F.pad(ego_feature, padding_pos)
        
        x = cav_feature
        for block in self.layers:
            x = block(ego_feature, x)
            
        x = sc_unpadding(x, padding_pos)
        return self.mlp_head(x)
    
    
class ConverterBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, heads, window_size, drop_out) -> None:
        super(ConverterBlock, self).__init__()
        
        dim_head =  input_dim // heads 
        
        self.window_size = window_size
        # self.inneratt = nn.Sequential(
        #     Rearrange(
        #         "b d (x w1) (y w2) -> b x y w1 w2 d", w1=window_size, w2=window_size
        #     ),
        #     PreNormResidual(
        #         input_dim, Attention(input_dim, dim_head, drop_out, window_size)
        #     ),
        #     PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
        #     Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
        # )
        
        self.fax_atten = nn.Sequential(
            # window attention, innner window
            Rearrange(
                "b d (x w1) (y w2) -> b x y w1 w2 d", w1=window_size, w2=window_size
            ),
            PreNormResidual(
                input_dim, Attention(input_dim, dim_head, drop_out, window_size)
            ),
            PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
            
            # grid attention, cross window
            Rearrange(
                "b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=window_size, w2=window_size
            ),
            PreNormResidual(
                input_dim, Attention(input_dim, dim_head, drop_out, window_size)
            ),
            PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
        )
        
        # self.cross_atten = \
        # CrossDomainSwapFusionBlock(input_dim, dim_head, heads, qkv_bias=False, win_size = window_size)
    
    def forward(self, x, pub_query = None):
        # x: b c h w
        # _, padding_pos = sc_padding(x, self.window_size)
        # x = F.pad(x, padding_pos)
        # pub_query = F.pad(pub_query, padding_pos)

        # x = self.inneratt(x)
        x = self.fax_atten(x)

        # x = self.cross_atten(x, pub_query)
        
        # unpad feature to origin size
        # x = sc_unpadding(x, padding_pos)
        
        return x 
    
class Converter_perdimlp(nn.Module):
    """
    Data rearrange -> swap block -> mlp_head
    """

    def __init__(self, args):
        super(Converter_perdimlp, self).__init__()

        self.layers = nn.ModuleList([])
        self.num_blocks = args["num_of_blocks"]

        # block related
        input_dim = args["dim"]
        mlp_dim = args["dim"]
        window_size = args["window_size"]
        drop_out = args["drop_out"]
        heads = args["heads"]

        self.window_size = args["window_size"]
        self.mask = False
# def __init__(self, input_dim, mlp_dim, heads, window_size, drop_out) 
        for i in range(self.num_blocks):
            block = ConverterBlock(input_dim, mlp_dim, heads, window_size, drop_out)
            self.layers.append(block)

        # mlp head, 默认初始化为1, 即无需调整
        self.mlp_head = nn.Parameter(torch.ones(input_dim))
        
        
    # def parameter_init(self):
        # self.mlp_head

    def forward(self, x):
        # if cannot be divided by window_size, pad the feature
        _, padding_pos = sc_padding(x, self.window_size)
        x = F.pad(x, padding_pos)

        for stage in self.layers:
            x = stage(x)
        
        # unpad feature to origin size
        x = sc_unpadding(x, padding_pos)

        return x * self.mlp_head.unsqueeze(1).unsqueeze(1)


class Converter(nn.Module):
    """
    Data rearrange -> swap block -> mlp_head
    """

    def __init__(self, args):
        super(Converter, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args["num_of_blocks"]

        # block related
        input_dim = args["dim"]
        mlp_dim = args["dim"]
        window_size = args["window_size"]
        drop_out = args["drop_out"]
        heads = args['heads']
        dim_head = input_dim // heads

        self.window_size = args["window_size"]
        self.mask = False

        for i in range(self.depth):
            block = SwapFusionBlock(input_dim, mlp_dim, dim_head, window_size, drop_out)
            self.layers.append(block)

        # mlp head
        self.mlp_head = nn.Sequential(
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange("b h w d -> b d h w"),
        )

    def forward(self, x, mask=None):
        # if cannot be divided by window_size, pad the feature
        _, padding_pos = sc_padding(x, self.window_size)
        x = F.pad(x, padding_pos)

        for stage in self.layers:
            x = stage(x, mask=mask)
        
        # unpad feature to origin size
        x = sc_unpadding(x, padding_pos)

        return self.mlp_head(x)


class CrossdomianConverter(nn.Module):
    def __init__(self, args):
        super(CrossdomianConverter, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args["num_of_blocks"]

        # block related
        input_dim = args["dim"]
        heads = args["heads"]
        # dim_head = args["dim_head"]
        dim_head = input_dim // heads
        window_size = args["window_size"]
        self.window_size = window_size

        for i in range(self.depth):
            self.layers.append(
                CrossDomainSwapFusionBlock(
                    input_dim, dim_head, heads, True, window_size
                )
            )

        # mlp head
        self.mlp_head = nn.Sequential(
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange("b h w d -> b d h w"),
        )

    def forward(self, ego_feature, cav_feature):
        _, padding_pos = sc_padding(ego_feature, self.window_size)
        ego_feature = F.pad(ego_feature, padding_pos)
        _, padding_pos = sc_padding(cav_feature, self.window_size)
        cav_feature = F.pad(cav_feature, padding_pos)

        x = cav_feature
        for block in self.layers:
            x = block(ego_feature, x)
            
        x = sc_unpadding(x, padding_pos)
        
        return self.mlp_head(x)

# if __name__ == "__main__":
#     import os

#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     ego = torch.rand(2, 256, 100, 352)  # .cuda()
#     cav = torch.rand(2, 256, 100, 352)  # .cuda()
#     h = 50
#     w = 176
#     # local attention
#     query = rearrange(
#         ego, "b d (x w1) (y w2) -> b x y w1 w2 d", x=h, w2=w
#     )  # window partition
#     key = rearrange(
#         cav, "b d (x w1) (y w2) -> b x y w1 w2 d", w1=h, w2=w
#     )  # window partition
#     print(query.shape)
#     print(key.shape)

#     args = {
#         "input_dim": 256,
#         "window_size": 8,
#         "dim_head": 32,
#         "heads": 16,
#         "depth": 1,
#     }
#     model = CrossDomainFusionEncoder(args)
#     output = model(ego, cav)
    # print(output)
