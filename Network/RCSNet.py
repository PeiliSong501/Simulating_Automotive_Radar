import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#range_image: H,W = 32,128
#local_pic: H,W = 128,128

class RCSNet(nn.Module):
    def __init__(self, batch_size, radarpoint_dim):
        super(RCSNet, self).__init__()
        self.batch_size = batch_size
        # Range Image layers
        self.range_conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.range_conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)
        self.range_conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)

        # Local Pic layers
        # self.local_conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
        self.local_conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        #self.local_conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)

        self.local_conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)
        self.local_conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        # Adjust to match dimensions
        self.adjust_conv = nn.Conv2d(16, 16, kernel_size=3, stride=(2, 2), padding=(2, 26))

        # Fusion and Reduction layers
        # self.fusion_conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.fusion_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.reduction_conv1 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)

        # rcs estimation
        self.fc_encode = nn.Linear(radarpoint_dim, 128)
        self.fc0 = nn.Linear(8 * 32, 128)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.adp_pool = nn.AdaptiveAvgPool2d((8, 32))

    def forward(self, batch_dict, current_batch_size):
        device = torch.device("cuda:3")
        # range_image = batch_dict['range_image'].to(device).float()  # [batch_size,32,128]
        local_pic = batch_dict['local_pic'].to(device).float()  # [batch_size,128,128,3]
        # radarpoint = batch_dict['radarpoint'].to(device).float()

        # range_image = torch.unsqueeze(range_image, 1)  # [batch_size,1,32,128]



        local_pic = local_pic.permute(0, 3, 1, 2)   #[batch_size,3,128,128]


        # range_x = F.relu(self.range_conv1(range_image))
        # range_x = F.relu(self.range_conv2(range_x))
        # range_x = F.relu(self.range_conv3(range_x))
        # range_x = self.adp_pool(range_x)

        local_x = F.relu(self.local_conv1(local_pic))
        local_x = F.relu(self.local_conv2(local_x))
        local_x = F.relu(self.local_conv3(local_x))

        local_x = F.relu(self.adjust_conv(local_x))
        local_x = self.adp_pool(local_x)

        # fused_x = torch.cat((range_x, local_x), dim=1)
        fused_x = local_x
        # fused_x = range_x

        fused_x = F.relu(self.fusion_conv1(fused_x))
        fused_x = F.relu(self.reduction_conv1(fused_x))
        fused_output = F.relu(self.final_conv(fused_x))
        fused_output = torch.squeeze(fused_output, 1)

        # size of fused_output: (batch_size, 8, 32)
        fused_output = fused_output.view(current_batch_size, -1)
        # radarpoint_feature = F.relu(self.fc_encode(radarpoint))
        fused_output = F.relu(self.fc0(fused_output))
        # combined_features = torch.cat((fused_output, radarpoint_feature), dim=1)

        x = fused_output

        #x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output