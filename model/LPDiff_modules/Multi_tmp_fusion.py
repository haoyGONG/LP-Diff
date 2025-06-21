import torch
import torch.nn as nn
import math
from torchvision import models
from torchvision.models.resnet import BasicBlock
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def kernel_size(in_channel): 
    """Calculate kernel size for 1D convolution as in ECA attention."""
    k = int((math.log2(in_channel) + 1) // 2)
    return k + 1 if k % 2 == 0 else k

class MultiScaleFeatureExtractor(nn.Module): 
    """
    Multi-scale feature extractor using three Conv-BN-ReLU blocks with downsampling.
    Input: (batch, 3, 128, 224); Output: (batch, 64, 16, 28)
    """
    def __init__(self, in_channel, out_channel):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class Encoder(nn.Module):
    """
    Simple CNN Encoder composed of sequential Conv-BN-ReLU blocks.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    """
    CNN Decoder with a sequence of ConvTranspose2d for upsampling.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class ResNetWithUpsample(nn.Module):
    """
    ResNet18 backbone (pretrained) with additional upsampling and channel adjustment layers.
    """
    def __init__(self, output_channels=3):
        super(ResNetWithUpsample, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.conv1x1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.conv1x1(x)
        x = self.upsample(x)
        return x

class FeatureFusion(nn.Module):
    """
    Fuse two feature maps by concatenation, followed by two conv layers and normalization.
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_channel)
        
    def forward(self, x, fuse):
        out = torch.cat((x, fuse), dim=1)
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.batchnorm(out)
        return out

class ChannelAttention(nn.Module):
    """
    Compute channel attention for two feature maps using global pooling and Conv1d.
    """
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.median_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(6, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(6, 1, kernel_size=self.k, padding=self.k // 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1, t2):
        t1_channel_avg_pool = self.avg_pool(t1)
        t2_channel_avg_pool = self.avg_pool(t2)
        t1_channel_max_pool = self.max_pool(t1)
        t2_channel_max_pool = self.max_pool(t2)
        t1_channel_median_pool = self.median_pool(t1)
        t2_channel_median_pool = self.median_pool(t2)
        channel_pool = torch.cat([
            t1_channel_avg_pool, t1_channel_max_pool, t1_channel_median_pool,
            t2_channel_avg_pool, t2_channel_max_pool, t2_channel_median_pool
        ], dim=2).squeeze(-1).transpose(1, 2)
        t1_channel_attention = self.channel_conv1(channel_pool)
        t2_channel_attention = self.channel_conv2(channel_pool)
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention], dim=0)
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)
        return channel_stack

class SpatialAttention(nn.Module):
    """
    Compute spatial attention for two feature maps using pooling and Conv2d.
    """
    def __init__(self):
        super().__init__()
        self.spatial_conv1 = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        self.spatial_conv2 = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1, t2):
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]
        t1_spatial_median_pool = torch.median(t1, dim=1, keepdim=True)[0]
        t2_spatial_median_pool = torch.median(t2, dim=1, keepdim=True)[0]
        spatial_pool = torch.cat([
            t1_spatial_avg_pool, t1_spatial_max_pool, t1_spatial_median_pool,
            t2_spatial_avg_pool, t2_spatial_max_pool, t2_spatial_median_pool
        ], dim=1)
        t1_spatial_attention = self.spatial_conv1(spatial_pool)
        t2_spatial_attention = self.spatial_conv2(spatial_pool)
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)
        spatial_stack = self.softmax(spatial_stack)
        return spatial_stack

class IntraframeAtt(nn.Module):
    """
    Intra-frame fusion attention module for temporal feature fusion.
    """
    def __init__(self, in_channel):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channel)
        self.spatial_attention = SpatialAttention()
        self.feature_fusion = FeatureFusion(in_channel*2, in_channel)

    def forward(self, t1, t2):
        channel_stack = self.channel_attention(t1, t2)
        spatial_stack = self.spatial_attention(t1, t2)
        stack_attention = channel_stack + spatial_stack + 1
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2
        return fuse

class InterframeAtt(nn.Module):
    """
    Inter-frame fusion attention using 1D conv over pooled statistics.
    """
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.median_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_conv = nn.Conv1d(3, 1, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, feature):
        feature_avg_pool = self.avg_pool(feature)
        feature_max_pool = self.max_pool(feature)
        feature_median_pool = self.median_pool(feature)
        feature_channel_cat = torch.cat([
            feature_avg_pool, feature_max_pool, feature_median_pool,
        ], dim=2).squeeze(-1).transpose(1, 2)
        feature_channel = self.channel_conv(feature_channel_cat)
        feature_channel = self.softmax(feature_channel).transpose(-1, -2).unsqueeze(-1)
        output = feature_channel * feature
        return output

class SobelModule(nn.Module):
    """
    Sobel filter module: applies Sobel x/y filters to each channel independently.
    """
    def __init__(self, in_channel=64, out_channel=192):
        super(SobelModule, self).__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_x.weight = nn.Parameter(sobel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y, requires_grad=False)
        self.softmax = nn.Softmax(dim=0)

    def apply_sobel(self, input):
        batch_size, channels, height, width = input.shape
        sobel_x_out = []
        sobel_y_out = []
        for c in range(channels):
            x_out = self.sobel_x(input[:, c:c+1, :, :])
            y_out = self.sobel_y(input[:, c:c+1, :, :])
            sobel_x_out.append(x_out)
            sobel_y_out.append(y_out)
        sobel_x_out = torch.cat(sobel_x_out, dim=1)
        sobel_y_out = torch.cat(sobel_y_out, dim=1)
        return sobel_x_out, sobel_y_out

    def forward(self, f1, f2):
        sobel_x_f1, sobel_y_f1 = self.apply_sobel(f1)
        sobel_x_f2, sobel_y_f2 = self.apply_sobel(f2)
        return sobel_x_f1, sobel_y_f1, sobel_x_f2, sobel_y_f2

class GradientSelfAttention(nn.Module):
    """
    Gradient-based self-attention using Sobel edges as attention weights.
    """
    def __init__(self, channels):
        super(GradientSelfAttention, self).__init__()
        self.channels = channels
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        sobel_kernel_x = torch.tensor([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[1, 2, 1],
                                       [0, 0, 0],
                                       [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_x = sobel_kernel_x.repeat(C, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y.repeat(C, 1, 1, 1)
        if x.is_cuda:
            sobel_kernel_x = sobel_kernel_x.cuda()
            sobel_kernel_y = sobel_kernel_y.cuda()
        grad_x = F.conv2d(x, sobel_kernel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_kernel_y, padding=1, groups=C)
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        attention_weights = torch.mean(grad_magnitude, dim=1, keepdim=True)
        attention_weights = torch.sigmoid(attention_weights)
        out = attention_weights * x
        out = self.gamma * out + x
        return out

class DirectionalConv(nn.Module):
    """
    Directional convolution: applies K convolutional kernels to features in different gradient directions.
    """
    def __init__(self, in_channels, out_channels, num_directions=8):
        super(DirectionalConv, self).__init__()
        self.num_directions = num_directions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(num_directions)
        ])
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        sobel_kernel_x = torch.tensor([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[1, 2, 1],
                                       [0, 0, 0],
                                       [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_x = sobel_kernel_x.repeat(C, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y.repeat(C, 1, 1, 1)
        if x.is_cuda:
            sobel_kernel_x = sobel_kernel_x.cuda()
            sobel_kernel_y = sobel_kernel_y.cuda()
        grad_x = F.conv2d(x, sobel_kernel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_kernel_y, padding=1, groups=C)
        theta = torch.atan2(grad_y, grad_x)
        theta_quantized = ((theta + np.pi) / (2 * np.pi) * self.num_directions).long() % self.num_directions
        out = torch.zeros(batch_size, self.out_channels, H, W).to(x.device)
        for k in range(self.num_directions):
            mask = (theta_quantized == k).float()
            F_k = x * mask
            F_k = self.convs[k](F_k)
            out += F_k
        out = out / self.num_directions
        return out

class TECrossAtt(nn.Module):
    """
    Multi-stage cross-attention block. Performs three stages of cross-attention, each using MultiheadAttention.
    """
    def __init__(self, embed_dim=64, num_heads=8):
        super(TECrossAtt, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj_a = nn.Linear(embed_dim, embed_dim)
        self.v_proj_b = nn.Linear(embed_dim, embed_dim)
        self.k_proj_c = nn.Linear(embed_dim, embed_dim)
        self.v_proj_d = nn.Linear(embed_dim, embed_dim)
        self.k_proj_e = nn.Linear(embed_dim, embed_dim)
        self.v_proj_f = nn.Linear(embed_dim, embed_dim)
        self.attn1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.attn2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.attn3 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1_q = nn.LayerNorm(embed_dim)
        self.norm1_k = nn.LayerNorm(embed_dim)
        self.norm1_v = nn.LayerNorm(embed_dim)
        self.norm1_out = nn.LayerNorm(embed_dim)
        self.norm2_q = nn.LayerNorm(embed_dim)
        self.norm2_k = nn.LayerNorm(embed_dim)
        self.norm2_v = nn.LayerNorm(embed_dim)
        self.norm2_out = nn.LayerNorm(embed_dim)
        self.norm3_q = nn.LayerNorm(embed_dim)
        self.norm3_k = nn.LayerNorm(embed_dim)
        self.norm3_v = nn.LayerNorm(embed_dim)
        self.norm3_out = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()

    def forward(self, a, b, c, d, e, f, g):
        batch_size, channels, height, width = g.size()
        seq_len = height * width
        def flatten(x):
            return x.view(batch_size, channels, -1).permute(0, 2, 1)
        def attention_block(q_input, k_input, v_input, attn_layer, norm_q, norm_k, norm_v, norm_out):
            q_norm = q_input
            k_norm = k_input
            v_norm = v_input
            q_norm = q_norm.transpose(0,1)
            k_norm = k_norm.transpose(0,1)
            v_norm = v_norm.transpose(0,1)
            attn_output, _ = attn_layer(q_norm, k_norm, v_norm)
            attn_output = attn_output + q_norm
            attn_output = attn_output.transpose(0,1)
            attn_output = self.activation(attn_output)
            attn_output = norm_out(attn_output)
            return attn_output
        q_input = flatten(g)
        q_input = self.q_proj(q_input)
        k1_input = flatten(a)
        k1_input = self.k_proj_a(k1_input)
        v1_input = flatten(b)
        v1_input = self.v_proj_b(v1_input)
        crossatt1 = attention_block(q_input, k1_input, v1_input, self.attn1, self.norm1_q, self.norm1_k, self.norm1_v, self.norm1_out)
        k2_input = flatten(c)
        k2_input = self.k_proj_c(k2_input)
        v2_input = flatten(d)
        v2_input = self.v_proj_d(v2_input)
        crossatt2 = attention_block(crossatt1, k2_input, v2_input, self.attn2, self.norm2_q, self.norm2_k, self.norm2_v, self.norm2_out)
        k3_input = flatten(e)
        k3_input = self.k_proj_e(k3_input)
        v3_input = flatten(f)
        v3_input = self.v_proj_f(v3_input)
        crossatt3 = attention_block(crossatt2, k3_input, v3_input, self.attn3, self.norm3_q, self.norm3_k, self.norm3_v, self.norm3_out)
        final_output = crossatt3.permute(0, 2, 1).view(batch_size, self.embed_dim, height, width)
        return final_output

class CrossAttentionLayer(nn.Module):
    """
    Standard transformer cross-attention layer with pre/post LayerNorm and feedforward block.
    """
    def __init__(self, d_model=64, num_heads=4, dim_feedforward=128):
        super(CrossAttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, f1, f2, mask=None):
        batch_size = f1.size(0)
        _, _, height, width = f1.size()
        f1 = f1.view(batch_size, self.d_model, -1).transpose(1, 2)
        f2 = f2.view(batch_size, self.d_model, -1).transpose(1, 2)
        q = self.q_linear(f2)
        k = self.k_linear(f1)
        v = self.v_linear(f1)
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)
        output = self.norm1(output + self.q_linear(f2))
        ff_output = self.fc2(F.relu(self.fc1(output)))
        output = self.norm2(ff_output + output)
        output = output.transpose(1, 2).view(batch_size, self.d_model, height, width)
        return output

class FeatureFusionModule(nn.Module):
    """
    Feature fusion module for fusing N features using convolution, BN, attention, and final fusion.
    """
    def __init__(self, input_channels, num_features=6):
        super(FeatureFusionModule, self).__init__()
        self.input_channels = input_channels
        self.num_features = num_features
        self.fuse_first_n = nn.Sequential(
            nn.Conv2d(in_channels=input_channels * num_features, out_channels=input_channels, kernel_size=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, *features):
        assert len(features) == self.num_features + 1, f"需要 {self.num_features + 1} 个特征图，但得到 {len(features)} 个。"
        x_n = torch.cat(features[:-1], dim=1)
        fused_n = self.fuse_first_n(x_n)
        attn_weights = self.attention_conv(features[-1])
        attn_fused = fused_n * attn_weights
        out = self.final_conv(attn_fused)
        return out

def set_parameter_requires_grad(model, trainable_layers=2):
    """
    Set requires_grad for layers of a model; freeze all but the last N layers.
    """
    layer_list = [
        model.conv1,
        model.bn1,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4
    ]
    num_layers = len(layer_list)
    num_frozen_layers = num_layers - trainable_layers
    for i, layer in enumerate(layer_list):
        if i < num_frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
    for param in model.conv1x1.parameters():
        param.requires_grad = True
    for param in model.bn2.parameters():
        param.requires_grad = True

class GradientCurvatureAttention(nn.Module):
    """
    Compute softmax-based attention using both first and second-order (curvature) gradients.
    """
    def __init__(self, alpha=0.5, beta=0.5):
        super(GradientCurvatureAttention, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        device = x.device
        sobel_kernel_x = torch.tensor([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]], dtype=torch.float32, device=device)
        sobel_kernel_y = torch.tensor([[1, 2, 1],
                                       [0, 0, 0],
                                       [-1, -2, -1]], dtype=torch.float32, device=device)
        sobel_kernel_x = sobel_kernel_x.unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = sobel_kernel_y.unsqueeze(0).unsqueeze(0)
        sobel_kernel_x = sobel_kernel_x.repeat(channels, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y.repeat(channels, 1, 1, 1)
        Gx = F.conv2d(x, sobel_kernel_x, padding=1, groups=channels)
        Gy = F.conv2d(x, sobel_kernel_y, padding=1, groups=channels)
        gradient_magnitude = torch.sqrt(Gx ** 2 + Gy ** 2 + 1e-6)
        kernel_xx = torch.tensor([[1, -2, 1],
                                  [2, -4, 2],
                                  [1, -2, 1]], dtype=torch.float32, device=device)
        kernel_yy = torch.tensor([[1, 2, 1],
                                  [-2, -4, -2],
                                  [1, 2, 1]], dtype=torch.float32, device=device)
        kernel_xy = torch.tensor([[-1, 0, 1],
                                  [0, 0, 0],
                                  [1, 0, -1]], dtype=torch.float32, device=device)
        kernel_xx = kernel_xx.unsqueeze(0).unsqueeze(0)
        kernel_yy = kernel_yy.unsqueeze(0).unsqueeze(0)
        kernel_xy = kernel_xy.unsqueeze(0).unsqueeze(0)
        kernel_xx = kernel_xx.repeat(channels, 1, 1, 1)
        kernel_yy = kernel_yy.repeat(channels, 1, 1, 1)
        kernel_xy = kernel_xy.repeat(channels, 1, 1, 1)
        Ixx = F.conv2d(x, kernel_xx, padding=1, groups=channels)
        Iyy = F.conv2d(x, kernel_yy, padding=1, groups=channels)
        Ixy = F.conv2d(x, kernel_xy, padding=1, groups=channels)
        epsilon = 1e-6
        Gx2 = Gx ** 2
        Gy2 = Gy ** 2
        numerator = Gx2 * Iyy - 2 * Gx * Gy * Ixy + Gy2 * Ixx
        denominator = (Gx2 + Gy2 + epsilon) ** 1.5
        curvature = numerator / denominator
        attention_weights = self.softmax(gradient_magnitude + curvature) + 1
        out = attention_weights * x
        return out

class MTA(nn.Module):
    """
    Main multi-module architecture, combining Encoder, cross-attention, gradient/curvature attention, and Decoder.
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        inner_channel = 3
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.kaiming_init(self.encoder)
        self.kaiming_init(self.decoder)
        self.GCA = GradientCurvatureAttention()
        self.IntraframeAtt = IntraframeAtt(inner_channel)
        self.kaiming_init(self.IntraframeAtt)
        self.crossatt = CrossAttentionLayer(d_model=64, num_heads=8, dim_feedforward=256)
        self.kaiming_init(self.crossatt)
        
    def forward(self, f1, f2, f3):
        f1_multi_scale = self.encoder(f1)
        f2_multi_scale = self.encoder(f2)
        f3_multi_scale = self.encoder(f3)
        crossatt_1 = self.crossatt(f1_multi_scale, f2_multi_scale) + f2_multi_scale
        crossatt_2 = self.crossatt(f2_multi_scale, f3_multi_scale) + f3_multi_scale
        GCA1 = self.GCA(crossatt_1)
        GCA2 = self.GCA(crossatt_2)
        channel_output = self.IntraframeAtt(GCA1, GCA2)
        return self.decoder(channel_output)
    
    def kaiming_init(self, model):
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias, 0)

if __name__ == '__main__':
    model = MTA(in_channel=3, out_channel=3)
    model = model.cuda()
    t1 = torch.randn(5, 3, 128, 224)
    t2 = torch.randn(5, 3, 128, 224)
    t3 = torch.randn(5, 3, 128, 224)
    t1 = t1.cuda()
    t2 = t2.cuda()
    t3 = t3.cuda()
    output = model(t1, t2, t3)
    print(output.shape)
