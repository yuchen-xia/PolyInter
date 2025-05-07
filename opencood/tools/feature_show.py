import torch
import matplotlib.pyplot as plt

def feature_show(feature, filename, type = 'mean', dim = None):
    """
    feature: [C, H, W]
    
    """
    # mean_tensor = torch.sum(feature,dim=0).cpu()
    if type == 'mean':
        mean_tensor = feature.mean(dim=0).cpu()
    if dim is not None:
        type == 'dim'
        mean_tensor = feature[dim].cpu()
    # print(torch.min(mean_tensor), torch.max(mean_tensor))
    # plt.clf()
    
    # 仅展示信息密度高的区域
    h_max, w_max = mean_tensor.size()  # 获取x轴的最大值  
    ax = plt.gca()  # 获取当前的轴  
    # left_margin = w_max * 0.35  # 计算左侧裁剪的宽度  
    # right_margin = w_max * 0.15  # 计算右侧裁剪的宽度 
    left_margin = w_max * 0.25  # 计算左侧裁剪的宽度  
    right_margin = w_max * 0.15  # 计算右侧裁剪的宽度   
    # ax.set_xlim([left_margin, w_max-right_margin])  # 设置x轴的范围以裁剪掉左右两侧的区域  
    
    # fig, _ = plt.subplots()
    # fig.set_size_inches(w_max, h_max)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    # 不显示坐标轴
    plt.axis('off')
    plt.Normalize(vmin=0, vmax=2)
    mean_tensor = mean_tensor.detach()
    plt.imshow(mean_tensor.numpy(), cmap='viridis')
    plt.savefig(filename, bbox_inches='tight', pad_inches = -0.1)

# tensor = torch.randn(100, 100, 128)
# filename = '/home/scz/hetecooper/hetecooper/test-1.png'
# feature_show(tensor, filename)
