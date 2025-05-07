import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.tools.feature_show import feature_show


class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.tau = args["tau"]
        self.max_voxel = args["max_voxel"]
        self.loss_dict = {}

    def forward(self, features_q, features_k, pos_region_ranges):
        """
        Parameters
        ----------
        features_q: b,c,h,w --> pub feature
        features_k: b,c,h,w --> local feature
        pos_region_ranges: b, max_num, h, w --> 指示每一对象在特征图中的mask
        output_dict : dict
        target_dict : dict
        """
        # b,c,h,w
        # features_q = output_dict["features_q"]
        # features_k = output_dict["features_k"]
        # (B,max_num,h,w)
        mask = pos_region_ranges

        device = features_q.device

        # max_num, b, 1, h, w
        pos_mask = mask.transpose(0, 1).contiguous().unsqueeze(2)
        # max_num, b, c, h, w, 在max_num维索引到的每个切片为某个对象单独的特征(其余特征均被置为0)
        masked_features_q = features_q * pos_mask.float()
        masked_features_k = features_k * pos_mask.float()
        # for i in range(masked_features_q.shape[0]):
        #     feature_show(masked_features_q[i][0], f'analysis/mask/q0_{i}.png')
        #     feature_show(masked_features_q[i][1], f'analysis/mask/q1_{i}.png')
        #     feature_show(masked_features_k[i][0], f'analysis/mask/k0_{i}.png')
        #     feature_show(masked_features_k[i][1], f'analysis/mask/k1_{i}.png')
        # print(masked_features_k.shape)
        # print(masked_features_q.shape)
        # (n,1,c)
        sampled_features_q, _ = self.sample_voxel(masked_features_q, mask, is_avg=True)
        # sampled_features_q_v2, _ = self.sample_voxel_v2(masked_features_q, mask, is_avg=True)
        # equal = torch.equal(sampled_features_q_v0, sampled_features_q)
        # print(equal)
        # print(torch.sum(sampled_features_q), torch.sum(sampled_features_q_v2))


        sampled_features_q = sampled_features_q.transpose(0, 1)
        # (n,p,c)
        sampled_features_k, pad_mask = self.sample_voxel(
            masked_features_k, mask, is_avg=True
        )
        # print("pad_mask", pad_mask.shape)
        # print("sampled_features_q", sampled_features_q.shape)
        # print('sampled_features_k',sampled_features_k.shape)

        norm_features_q = F.normalize(sampled_features_q, p=2, dim=-1)
        norm_features_k = F.normalize(sampled_features_k, p=2, dim=-1)

        # n,p,n
        sim = norm_features_k @ norm_features_q.transpose(-1, -2)
        # print("sim", sim.shape)

        logits = sim.clone()
        logits /= self.tau
        
        # label设置的目标是实现分类, 对其到某几个标定类别的值就可
        labels = (
            torch.arange(logits.shape[0], device=device)
            .unsqueeze(-1)
            .expand(logits.shape[0], logits.shape[1])
        )
        # 计算交叉熵损失
        loss = F.cross_entropy(logits[pad_mask], labels[pad_mask])

        target_idx = [*range(len(sim))]
        target_idx_ = torch.zeros_like(sim)
        target_idx_[target_idx, :, target_idx] = 1.0
        target_idx = target_idx_.bool()

        pos_cos_sim = sim[target_idx].mean()
        neg_cos_sim = (
            sim[~target_idx].mean()
            if sampled_features_k.shape[0] > 1
            else torch.tensor(0).to(device)
        )

        sim = sim.softmax(-1)
        pos_softmax_sim = sim[target_idx].mean()
        neg_softmax_sim = (
            sim[~target_idx].mean()
            if sampled_features_k.shape[0] > 1
            else torch.tensor(0).to(device)
        )
        # print(loss, pos_cos_sim, neg_cos_sim)
        # exit(0)

        self.loss_dict.update(
            {
                "loss": loss,
                "pos_cos_sim": pos_cos_sim,
                "neg_cos_sim": neg_cos_sim,
                "pos_softmax_sim": pos_softmax_sim,
                "neg_softmax_sim": neg_softmax_sim,
            }
        )

        return loss


    def sample_voxel(self, feature, mask, is_avg):
        """
        mask: (B,max_num,h,w)
        feature:(max_num,B,c,h,w)
        
        return: 
            sampled_voxel: num_obj(max_num * B) , 1, c; 该样本的平均值
            pad_list: num_obj(max_num * B); 该样本是否有效
        """
        # 真的没问题吗？
        mask = mask.flatten(0, 1)
        # N,c,h,w
        feature = feature.flatten(0, 1)
        N = feature.shape[0]
        f_list = []
        pad_list = []
        for i in range(N):
            index = torch.stack(torch.where(mask[i] == True)) # 找到值为非0的特征点, torch.where() 返回索引的list, 分别index每个维度
            # print(len(index))
            if index.shape[1] == 0:
                continue
            # 随机重排索引h和w的index, 以从该obj中采样出max_voxel个特征点(使用随机是为了使数据分布更加均匀)
            idx = torch.randperm(index.shape[1])
            index = index[:, idx].view(index.size())
            # sample positive region voxel, n,c,h,w -> c, h * w -> h * w, c 
            sampled_voxel = feature[i, :, index[0], index[1]].transpose(0, 1)[ # 点阵索引时, 将索引出来的点排列为一个新的位置
                : self.max_voxel
            ]
            if is_avg:
                sampled_voxel = torch.mean(sampled_voxel, dim=(0))
                pad = sampled_voxel[0].bool()
                pad = pad.unsqueeze(0)
                sampled_voxel = sampled_voxel.unsqueeze(0)
            else:
                sampled_voxel = F.pad(
                    sampled_voxel, (0, 0, 0, self.max_voxel - sampled_voxel.shape[0])
                )
                pad = sampled_voxel[:, 0].bool()
            f_list.extend([sampled_voxel])
            pad_list.extend([pad])
        print(torch.stack(f_list).shape)
        print(torch.stack(pad_list).shape)
        return torch.stack(f_list), torch.stack(pad_list)

    def temperature_cosinesim(self, q, k):
        w = q @ k.transpose(-1, -2)
        # divide the dot product by the temperature
        w_ = w / self.tau
        return w_

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict["loss"]
        pos_cos_sim = self.loss_dict["pos_cos_sim"]
        neg_cos_sim = self.loss_dict["neg_cos_sim"]
        pos_softmax_sim = self.loss_dict["pos_softmax_sim"]
        neg_softmax_sim = self.loss_dict["neg_softmax_sim"]
        if pbar is None:
            print(
                "[epoch %d] || Loss: %.4f || pos_sim: %.4f || neg_sim: %.4f||"
                % (
                    epoch,
                    total_loss.item(),
                    pos_cos_sim.item(),
                    neg_cos_sim.item(),
                )
            )
        else:
            pbar.set_description(
                "[epoch %d] || Loss: %.4f || pos_sim: %.4f || neg_sim: %.4f||"
                % (
                    epoch,
                    total_loss.item(),
                    pos_cos_sim.item(),
                    neg_cos_sim.item(),
                )
            )

        writer.add_scalar("pos_avg_cos_sim", pos_cos_sim, epoch * batch_len + batch_id)
        writer.add_scalar("neg_avg_cos_sim", neg_cos_sim, epoch * batch_len + batch_id)
        writer.add_scalar(
            "pos_softmax_sim", pos_softmax_sim, epoch * batch_len + batch_id
        )
        writer.add_scalar(
            "neg_softmax_sim", neg_softmax_sim, epoch * batch_len + batch_id
        )


if __name__ == "__main__":
    
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ego = torch.rand(4, 256, 50, 176)  # .cuda()
    cav = torch.rand(4, 256, 50, 176)  # .cuda()
    mask = torch.rand(4, 50, 50, 176)>0.2
    
    # print(mask[:,:,[1,2],[1,2]].shape)
    
    args ={
        'tau': 0.1,
        'max_voxel': 40
    }
    data_dict = {"features_q": ego, "features_k": cav}
    target_dict = {"pos_region_ranges": mask}
    model = ContrastiveLoss(args)
    output = model(ego, cav, mask)
    print(output.shape)