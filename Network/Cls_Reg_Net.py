import torch
import torch.nn as nn
import torch.nn.functional as F

def save_backbone_params(backbone, path):
    torch.save(backbone.state_dict(), path)

def load_backbone_params(backbone, path):
    state_dict = torch.load(path)
    backbone.load_state_dict(state_dict)

class CrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.out_channels = out_channels
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, img_feat, lidar_feat):
        batch_size, channels, width, height = img_feat.size()

        # Compute queries, keys, and values
        query_img = self.query_conv(img_feat).view(batch_size, -1, width * height).permute(0, 2,
                                                                                           1)  # shape: (batch_size, width*height, out_channels)
        key_lidar = self.key_conv(lidar_feat).view(batch_size, -1, width * height).permute(0, 2,
                                                                                           1)  # shape: (batch_size, width*height, out_channels)
        value_lidar = self.value_conv(lidar_feat).view(batch_size, -1,
                                                       width * height)  # shape: (batch_size, out_channels, width*height)

        # Compute attention scores
        attention1 = torch.bmm(query_img, key_lidar.permute(0, 2, 1))  # shape: (batch_size, width*height, width*height)
        attention1 = self.softmax(attention1)
        attention1 = self.dropout(attention1)

        # Compute cross-attended features
        cross_feat1 = torch.bmm(attention1,
                               value_lidar.permute(0, 2, 1))  # shape: (batch_size, width*height, out_channels)
        cross_feat1 = cross_feat1.permute(0, 2, 1).view(batch_size, self.out_channels, width,
                                                      height)  # shape: (batch_size, out_channels, width, height)


        return cross_feat1


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.i_conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )

        self.i_conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )

        self.i_conv3 = nn.Sequential(
            nn.Conv2d(16, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )

        self.l_conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )

        self.l_conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )

        self.l_conv3 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )

        # self.fusion = nn.Sequential(
        #     nn.Conv2d(8, 4, 3, 1, 1),
        #     nn.BatchNorm2d(4),
        #     nn.LeakyReLU(),
        #     #nn.MaxPool2d(2),
        # )

        self.cross_attention = CrossAttention(4,4)
        self.dropout = nn.Dropout(p=0.5)



    def forward(self, image, lidar, current_bs):

        # image = batch_dict['local_pic'].cuda().float()
        # image = image.permute(0, 3, 1, 2)
        image = self.i_conv1(image)
        image = self.i_conv2(image)
        image = self.i_conv3(image)


        # lidar = batch_dict['local_pcl'].cuda().float()
        # lidar = torch.unsqueeze(lidar,1)

        lidar = self.l_conv1(lidar)
        lidar = self.l_conv2(lidar)
        lidar = self.l_conv3(lidar)

        # fused_f = torch.cat((image, lidar), dim=1)
        # fused_f = self.dropout(fused_f)
        # fused_f = self.fusion(fused_f)
        # fused_f = fused_f.reshape(current_bs, -1)
        #
        # return fused_f

        cross_feat = self.cross_attention(image, lidar)

        cross_feat = cross_feat.reshape(current_bs, -1)
        cross_feat = self.dropout(cross_feat)

        return cross_feat


class ClassificationBranch(nn.Module):
    def __init__(self, backbone):
        super(ClassificationBranch, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, 2)
        self.fc_velo_pos = nn.Linear(7,64)
        #self.fc_res = nn.Linear(128,64)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, batch_dict, current_bs):
        # velo_pos = batch_dict['velo_pos'].cuda().float()
        # velo_pos = self.dropout(velo_pos)
        # velo_pos = self.fc_velo_pos(velo_pos)


        image = batch_dict['local_pic'].cuda().float()
        image = image.permute(0, 3, 1, 2)
        lidar = batch_dict['local_pcl'].cuda().float()
        lidar = torch.unsqueeze(lidar, 1)
        combined_feature = self.backbone(image, lidar, current_bs)
        combined_feature = self.dropout(combined_feature)
        out = F.relu(self.fc1(combined_feature))
        out = self.dropout(out)

        # out1 = torch.cat((out, velo_pos),dim = 1)
        # out1 = self.dropout(out1)
        # out1 = self.fc_res(out1)
        # out += out1
        out = F.relu(self.fc2(out))

        return out


class RegressionBranch(nn.Module):
    def __init__(self, backbone):
        super(RegressionBranch, self).__init__()
        self.backbone = backbone
        self.fc_feature = nn.Sequential(
            nn.Linear(576, 32),
            nn.ReLU(),
        )
        self.fc_radar = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.5),
        )
        self.fc_ri = nn.Linear(4*8*32,64)

    def forward(self, batch_dict, current_bs):
        range_image = batch_dict['range_image'].cuda().float()
        range_image = torch.unsqueeze(range_image, 1)
        range_image = self.conv1(range_image)
        range_image = self.conv2(range_image)
        range_image = self.conv3(range_image)
        range_image = range_image.reshape(current_bs, -1)
        range_image = self.fc_ri(range_image)

        radar_input = batch_dict['radarpoint'].cuda().float()
        image = batch_dict['local_pic'].cuda().float()
        image = image.permute(0, 3, 1, 2)
        lidar = batch_dict['local_pcl'].cuda().float()
        lidar = torch.unsqueeze(lidar, 1)
        combined_feature = self.backbone(image, lidar, current_bs)
        combined_feature = self.dropout(combined_feature)

        feature = torch.flatten(combined_feature, start_dim=1)
        feature = self.fc_feature(feature)
        radar = self.fc_radar(radar_input)
        combined_feature = torch.cat((feature, radar), dim=1)
        combined_feature = torch.cat((combined_feature,range_image),dim = 1)
        out = self.fc_combined(combined_feature)
        return out



