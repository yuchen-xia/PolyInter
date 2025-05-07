


from opencood.models.fuse_modules.fusion_in_one import AttFusion, CoBEVT, DiscoFusion, MaxFusion, V2VNetFusion, V2XViTFusion, Where2commFusion, Who2comFusion
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion


def build_fusion_net(fuion_net_args):
    method = fuion_net_args['method']
    net_args = fuion_net_args['args']
    
    assert method in ["fcooper", 'att', 'disconet', 'v2vnet', 'v2xvit', 'cobevt',\
                      'where2comm', 'who2com', 'pyramid']
    
    if method  == "fcooper":
        fusion_net = MaxFusion()
    if method  == "att":
        fusion_net = AttFusion(net_args['feat_dim'])
    if method  == "disconet":
        fusion_net = DiscoFusion(net_args)
    if method  == "v2vnet":
        fusion_net = V2VNetFusion(net_args)
    if method  == 'v2xvit':
        fusion_net = V2XViTFusion(net_args)
    if method  == 'cobevt':
        fusion_net = CoBEVT(net_args)
    if method  == 'where2comm':
        fusion_net = Where2commFusion(net_args['feat_dim'])
    if method  == 'who2com':
        fusion_net = Who2comFusion(net_args)
    if method == 'pyramid':
        fusion_net = PyramidFusion(net_args)
        
        
    return fusion_net