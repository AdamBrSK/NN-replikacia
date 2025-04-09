import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import os

class ImBottleneck1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=None, reduction=2):
        super(ImBottleneck1D, self).__init__()
        mid_channels = out_channels // reduction
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        if dilation_rates is None:
            self.conv2_1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3,1), padding=(1,0), bias=False)
            self.conv2_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,3), padding=(0,1), bias=False)
        else:
            self.conv2_1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3,1), 
                                    padding=(dilation_rates[0],0), dilation=(dilation_rates[0],1), bias=False)
            self.conv2_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,3), 
                                    padding=(0,dilation_rates[0]), dilation=(1,dilation_rates[0]), bias=False)
        
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        if dilation_rates is None:
            self.conv3_1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3,1), padding=(1,0), bias=False)
            self.conv3_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,3), padding=(0,1), bias=False)
        else:
            self.conv3_1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3,1), 
                                    padding=(dilation_rates[1],0), dilation=(dilation_rates[1],1), bias=False)
            self.conv3_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1,3), 
                                    padding=(0,dilation_rates[1]), dilation=(1,dilation_rates[1]), bias=False)
        
        self.bn3 = nn.BatchNorm2d(mid_channels)
        
        self.conv4 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.bn3(out)
        out = self.relu(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ChannelMLP(nn.Module):
    def __init__(self, channels, hidden_dim=None):
        super(ChannelMLP, self).__init__()
        hidden_dim = hidden_dim or channels * 2
        
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        return x

class SpatialMLP(nn.Module):
    def __init__(self, channels, hidden_dim=None):
        super(SpatialMLP, self).__init__()
        hidden_dim = hidden_dim or channels * 2
        
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class HybridMLPBlock(nn.Module):
    def __init__(self, channels, height, width):
        super(HybridMLPBlock, self).__init__()
        self.channel_mlp = ChannelMLP(channels)
        self.spatial_mlp = SpatialMLP(channels)
        
    def forward(self, x):
        x = x + self.channel_mlp(x)
        x = x + self.spatial_mlp(x)
        return x

class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pool_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.adjust_channels = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        conv_out = self.conv_path(x)
        pool_out = self.pool_path(x)
        combined = torch.cat([conv_out, pool_out], dim=1)
        return self.adjust_channels(combined)

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSamplingBlock, self).__init__()
        
        self.bilinear = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.transposed = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.reduce = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        bilinear_out = self.bilinear(x)
        transposed_out = self.transposed(x)
        combined = torch.cat([bilinear_out, transposed_out], dim=1)
        return self.reduce(combined)

class LaneDetectionNetwork(nn.Module):
    def __init__(self, num_classes=2, num_lanes=4):
        super(LaneDetectionNetwork, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.encoder = nn.ModuleList([
            DownSamplingBlock(16, 32),
            *[ImBottleneck1D(32, 32) for _ in range(4)],
            DownSamplingBlock(32, 64),
            *[ImBottleneck1D(64, 64, dilation_rates=[2,3]) for _ in range(4)],
            DownSamplingBlock(64, 128),
            *[ImBottleneck1D(128, 128, dilation_rates=[3,5]) for _ in range(5)],
            HybridMLPBlock(128, 64, 32)
        ])
        
        self.lane_existence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_lanes),
            nn.Sigmoid()
        )
        
        self.decoder = nn.ModuleList([
            UpSamplingBlock(128, 64),
            *[ImBottleneck1D(64, 64) for _ in range(2)],
            UpSamplingBlock(64, 32),
            *[ImBottleneck1D(32, 32) for _ in range(2)],
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        ])
        
        self.skip_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.initial_conv(x)
        
        skip_connection = None
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i == 5:
                skip_connection = self.skip_conv(x)
        
        existence = self.lane_existence(x)
        
        for i, layer in enumerate(self.decoder):
            if i == 0:
                x = layer(x)
                x = x + skip_connection
            else:
                x = layer(x)
        
        segmentation = torch.sigmoid(x)
        
        return segmentation, existence

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]))
        
    def dice_loss(self, pred, target):
        smooth = 1.0
        pred = pred[:, 1]
        target = target.float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        return 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    
    def bce_loss(self, pred, target):
        return F.binary_cross_entropy(pred, target.float())
    
    def forward(self, pred_seg, pred_exist, target_seg, target_exist):
        ce = self.ce_loss(pred_seg, target_seg)
        dice = self.dice_loss(pred_seg, target_seg)
        exist = self.bce_loss(pred_exist, target_exist)
        
        return self.alpha * ce + self.beta * dice + self.gamma * exist

def preprocess_image(image_path, target_size=(512, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    return image, image_tensor

def preprocess_mask(mask_path, target_size=(512, 256)):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 128).astype(np.uint8)
    mask_tensor = torch.from_numpy(mask).long()
    return mask_tensor

def train_model(model, image_tensor, mask_tensor, device, num_epochs=100):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = CombinedLoss(alpha=1.0, beta=0.5, gamma=0.1)
    
    image_tensor = image_tensor.unsqueeze(0).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)
    
    target_exist = torch.ones((1, 4), device=device)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        seg_out, exist_out = model(image_tensor)
        
        loss = criterion(seg_out, exist_out, mask_tensor, target_exist)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def process_image(model, image, image_tensor, device, output_path='output_with_lanes.jpg', target_size=(512, 256)):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        seg_out, _ = model(image_tensor)
        seg_out = seg_out.squeeze(0).cpu().numpy()
        lane_mask = seg_out[1]
        
        lane_mask_binary = (lane_mask > 0.5).astype(np.uint8) * 255
        
        kernel = np.ones((3, 3), np.uint8)
        lane_mask_cleaned = cv2.morphologyEx(lane_mask_binary, cv2.MORPH_OPEN, kernel)
        
        lane_mask_thick = cv2.dilate(lane_mask_cleaned, kernel, iterations=2)
        
        lane_overlay = np.zeros_like(image)
        lane_overlay[lane_mask_thick == 255] = [0, 255, 0]
        final_image = cv2.addWeighted(image, 0.8, lane_overlay, 0.7, 0)
        
        #cv2.imwrite(output_path, final_image)
        #cv2.imshow('Detected Lanes', final_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print(f"Image with green lane lines saved as '{output_path}'")

if __name__ == "__main__":
    model = LaneDetectionNetwork(num_classes=2, num_lanes=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    image_path = './10.jpg'
    mask_path = './20.png'
    
    target_size = (512, 256)
    original_image, image_tensor = preprocess_image(image_path, target_size)
    mask_tensor = preprocess_mask(mask_path, target_size)
    
    print("Starting training...")
    train_model(model, image_tensor, mask_tensor, device, num_epochs=500)
    
    print("Processing image...")
    process_image(model, original_image, image_tensor, device, target_size=target_size)