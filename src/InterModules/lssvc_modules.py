import torch
import torch.nn as nn

from src.InterModules.video_net_component import GDN, ResBlock, flow_warp, bilinearupsacling, subpel_conv3x3, \
    subpel_conv1x1
from torch.nn.functional import interpolate

g_ch_1x = 48
g_ch_2x = 64
g_ch_4x = 96
g_ch_8x = 96
g_ch_16x = 128


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope=0.01, inplace=False):
        super().__init__()
        dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, dw_ch, 1, stride=stride),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(dw_ch, dw_ch, depth_kernel, padding=depth_kernel // 2,
                                    groups=dw_ch)
        self.conv2 = nn.Conv2d(dw_ch, out_ch, 1)

        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        elif in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity


class ConvFFN(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        internal_ch = max(min(in_ch * 4, 1024), in_ch * 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, internal_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            nn.Conv2d(internal_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )

    def forward(self, x):
        identity = x
        return identity + self.conv(x)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1,
                 slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride, slope=slope_depth_conv, inplace=inplace),
            ConvFFN(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class OffsetDiversity(nn.Module):
    def __init__(self, in_channel=g_ch_1x, aux_feature_num=g_ch_1x + 3 + 2,
                 offset_num=2, group_num=16, max_residue_magnitude=40, inplace=False):
        super().__init__()
        self.in_channel = in_channel
        self.offset_num = offset_num
        self.group_num = group_num
        self.max_residue_magnitude = max_residue_magnitude
        self.conv_offset = nn.Sequential(
            nn.Conv2d(aux_feature_num, g_ch_2x, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, g_ch_2x, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, 3 * group_num * offset_num, 3, 1, 1),
        )
        self.fusion = nn.Conv2d(in_channel * offset_num, in_channel, 1, 1, groups=group_num)

    def forward(self, x, aux_feature, flow):
        B, C, H, W = x.shape
        out = self.conv_offset(aux_feature)
        out = bilinearupsacling(out)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        mask = torch.sigmoid(mask)
        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.repeat(1, self.group_num * self.offset_num, 1, 1)

        # warp
        offset = offset.view(B * self.group_num * self.offset_num, 2, H, W)
        mask = mask.view(B * self.group_num * self.offset_num, 1, H, W)
        x = x.view(B * self.group_num, C // self.group_num, H, W)
        x = x.repeat(self.offset_num, 1, 1, 1)
        x = flow_warp(x, offset)
        x = x * mask
        x = x.view(B, C * self.offset_num, H, W)
        x = self.fusion(x)

        return x


class HybridWeightGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator1 = nn.Sequential(
            nn.Conv2d(g_ch_1x * 2, 64, 3, stride=1, padding=1),
            ResBlock(64, end_with_relu=True),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )
        self.generator2 = nn.Sequential(
            nn.Conv2d(g_ch_2x * 2, 64, 3, stride=1, padding=1),
            ResBlock(64, end_with_relu=True),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )
        self.generator3 = nn.Sequential(
            nn.Conv2d(g_ch_4x * 2, 64, 3, stride=1, padding=1),
            ResBlock(64, end_with_relu=True),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )
        self.activate1 = nn.Softmax(dim=1)
        self.activate2 = nn.Softmax(dim=1)
        self.activate3 = nn.Softmax(dim=1)

    def forward(self, ctx_temp, ctx_spat):
        if ctx_spat is None:
            return [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        feature1 = torch.cat([ctx_temp[0], ctx_spat[0]], dim=1)
        map_feature1 = self.generator1(feature1)
        weight_map = self.activate1(map_feature1)
        map_temp1, map_spat1 = weight_map.chunk(2, 1)

        feature2 = torch.cat([ctx_temp[1], ctx_spat[1]], dim=1)
        map_feature2 = self.generator2(feature2)
        weight_map = self.activate2(map_feature2)
        map_temp2, map_spat2 = weight_map.chunk(2, 1)

        feature3 = torch.cat([ctx_temp[2], ctx_spat[2]], dim=1)
        map_feature3 = self.generator3(feature3)
        weight_map = self.activate1(map_feature3)
        map_temp3, map_spat3 = weight_map.chunk(2, 1)
        return [map_temp1, map_temp2, map_temp3], [map_spat1, map_spat2, map_spat3]


class TextureExtractor(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(64, g_ch_1x, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_1x, g_ch_2x, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_2x, g_ch_4x, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(g_ch_4x, inplace=inplace)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class FeatureExtractor(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_1x, g_ch_2x, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_2x, g_ch_4x, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(g_ch_4x, inplace=inplace)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv3_up = subpel_conv3x3(g_ch_4x, g_ch_2x, 2)
        self.res_block3_up = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3_out = nn.Conv2d(g_ch_4x, g_ch_4x, 3, padding=1)
        self.res_block3_out = ResBlock(g_ch_4x, inplace=inplace)
        self.conv2_up = subpel_conv3x3(g_ch_2x * 2, g_ch_1x, 2)
        self.res_block2_up = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2_out = nn.Conv2d(g_ch_2x * 2, g_ch_2x, 3, padding=1)
        self.res_block2_out = ResBlock(g_ch_2x, inplace=inplace)
        self.conv1_out = nn.Conv2d(g_ch_1x * 2, g_ch_1x, 3, padding=1)
        self.res_block1_out = ResBlock(g_ch_1x, inplace=inplace)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out

        return context1, context2, context3


class ResEncoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x + 3, g_ch_2x, 3, stride=2, padding=1)
        self.res1 = ResBlock(g_ch_2x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_2x * 2, g_ch_4x, 3, stride=2, padding=1)
        self.res2 = ResBlock(g_ch_4x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_4x * 2, g_ch_8x, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ResDecoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.up1 = subpel_conv3x3(g_ch_16x, g_ch_8x, 2)
        self.up2 = subpel_conv3x3(g_ch_8x, g_ch_4x, 2)
        self.res1 = ResBlock(g_ch_4x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up3 = subpel_conv3x3(g_ch_4x * 2, g_ch_2x, 2)
        self.res2 = ResBlock(g_ch_2x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up4 = subpel_conv3x3(g_ch_2x * 2, 32, 2)

    def forward(self, x, context2, context3):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=g_ch_1x, res_channel=32, inplace=False):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, g_ch_1x, 3, stride=1, padding=1)
        self.unet_1 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.unet_2 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.recon_conv = nn.Conv2d(g_ch_1x, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class UNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock(64, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


class MvResampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1)
        )
        self.feature_refine = nn.Sequential(
            DepthConvBlock(64, 64, inplace=False),
            DepthConvBlock(64, 64, inplace=False),
        )
        self.recon_conv = nn.Conv2d(64, 2, 3, stride=1, padding=1)

    def forward(self, mv_bl, shape_hr, s):
        mv_feature = self.conv1(mv_bl)
        mv_up_feature = interpolate(mv_feature, size=shape_hr, mode='bilinear', align_corners=False)
        mv_up_feature = self.conv2(mv_up_feature)
        mv_refine_feature = self.feature_refine(mv_up_feature)
        mv_feature = mv_refine_feature + mv_up_feature
        mv = self.recon_conv(mv_feature)
        return s * mv


class TextureResampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_adaptor = nn.ModuleDict({
            'base_layer_adaptor': nn.Conv2d(64, 64, 3, stride=1, padding=1),
            'enhance_layer_adaptor': nn.Conv2d(g_ch_1x, 64, 3, stride=1, padding=1)
        })
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1)
        )
        self.feature_refine = nn.Sequential(
            DepthConvBlock(64, 64, inplace=False),
            DepthConvBlock(64, 64, inplace=False),
        )

    def forward(self, texture_bl, shape_hr):
        feature = self.conv_adaptor['base_layer_adaptor'](texture_bl) if texture_bl.shape[1] == 64 else self.conv_adaptor['enhance_layer_adaptor'](texture_bl)
        feature = self.conv1(feature)
        up_feature = interpolate(feature, size=shape_hr, mode='bilinear', align_corners=False)
        up_feature = self.conv2(up_feature)
        refine_feature = self.feature_refine(up_feature)
        feature = refine_feature + up_feature
        return feature


class LayerPriorResampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_adaptor = nn.ModuleDict({
            'base_layer_adaptor': nn.Conv2d(96, 96, 3, stride=1, padding=1),
            'enhance_layer_adaptor': nn.Conv2d(g_ch_16x, 96, 3, stride=1, padding=1)
        })
        self.conv1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, 3, stride=1, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(96, g_ch_16x, 3, stride=1, padding=1)
        )
        self.feature_refine = nn.Sequential(
            DepthConvBlock(g_ch_16x, g_ch_16x, inplace=False),
            DepthConvBlock(g_ch_16x, g_ch_16x, inplace=False),
        )

    def forward(self, y_hat_bl, shape_hr):
        feature = self.conv_adaptor['base_layer_adaptor'](y_hat_bl) if y_hat_bl.shape[1] == 96 else self.conv_adaptor['enhance_layer_adaptor'](y_hat_bl)
        feature = self.conv1(feature)
        up_feature = interpolate(feature, size=shape_hr, mode='bilinear', align_corners=False)
        up_feature = self.conv2(up_feature)
        refine_feature = self.feature_refine(up_feature)
        layer_prior = refine_feature + up_feature
        return layer_prior


class PriorFusion(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.prior_fusion_conv = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 2, inplace=inplace),
        )

    def forward(self, hyper_prior, temporal_prior, layer_prior):
        params = self.prior_fusion_conv(torch.cat([hyper_prior, temporal_prior, layer_prior], dim=1))
        return params


class MVResEncoder(nn.Module):
    def __init__(self, channel_mv=64):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(2, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            ResBlock(channel_mv, start_from_relu=False),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(channel_mv + channel_mv, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            ResBlock(channel_mv, start_from_relu=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            ResBlock(channel_mv, start_from_relu=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
        )

    def forward(self, mv, mv_ctx=None):
        feature = self.encoder1(mv)
        mv_y = self.encoder2(torch.cat([feature, mv_ctx], dim=1))
        return mv_y


class MVResDecoder(nn.Module):
    def __init__(self, channel_mv=64):
        super().__init__()
        self.decoder1 = nn.Sequential(
            subpel_conv3x3(channel_mv, channel_mv, r=2),
            nn.LeakyReLU(negative_slope=0.1),
            ResBlock(channel_mv, start_from_relu=False),
            GDN(channel_mv, inverse=True),
            subpel_conv3x3(channel_mv, channel_mv, r=2),
            GDN(channel_mv, inverse=True),
            subpel_conv3x3(channel_mv, channel_mv, r=2),
            GDN(channel_mv, inverse=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(channel_mv + channel_mv, channel_mv, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            subpel_conv3x3(channel_mv, 2, r=2)
        )

    def forward(self, mv_y_hat, mv_ctx=None):
        feature = self.decoder1(mv_y_hat)
        mv_hat = self.decoder2(torch.cat([feature, mv_ctx], dim=1))
        return mv_hat


class MVContextTransformer(nn.Module):
    def __init__(self, channel_mv=64):
        super().__init__()

        self.transform = nn.Sequential(
            nn.Conv2d(2, channel_mv, 3, stride=2, padding=1),
            ResBlock(channel_mv, start_from_relu=True),
        )

    def forward(self, mv_upsample):
        mv_ctx = self.transform(mv_upsample)
        return mv_ctx
