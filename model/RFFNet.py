import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from .arch_util import LayerNorm2d, DownSample, SkipUpSample, DFFN, DropPath, SAM, ResBlock, conv
from einops import rearrange

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.conv2_1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        f_cat = torch.cat([x1, x2], dim=1)
        feats = self.relu(self.conv1(f_cat))
        attn1 = self.sigmoid(self.conv2_1(feats))
        attn2 = self.sigmoid(self.conv2_2(feats))
        return x1*attn1, x2*attn2

# Multi-scale Consistency Calibration Module
class MICM(nn.Module):
    def __init__(self, dim, bias):
        super(MICM, self).__init__()
        self.in_x = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.in_y = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.pools_sizes = [8,4,2]
        pools_x, pools_y, convs_x, convs_y, attns = [],[],[],[],[]
        for i in self.pools_sizes:
            pools_x.append(nn.AvgPool2d(kernel_size=i, stride=i))
            pools_y.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs_x.append(nn.Conv2d(dim, dim, 3, 1, 1, bias=bias))
            convs_y.append(nn.Conv2d(dim, dim, 3, 1, 1, bias=bias))
            attns.append(SpatialAttention(dim))
        self.pools_x = nn.ModuleList(pools_x)
        self.pools_y = nn.ModuleList(pools_y)
        self.convs_x = nn.ModuleList(convs_x)
        self.convs_y = nn.ModuleList(convs_y)
        self.attns = nn.ModuleList(attns)
        self.relu = nn.GELU()
        self.sum_x = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)
        self.sum_y = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias)

    def forward(self, x, y):
        x_size = x.size()
        res_x = self.in_x(x)
        res_y = self.in_y(y)
        for i in range(len(self.pools_sizes)):
            if i == 0:
                x_, y_ = self.attns[i](self.convs_x[i](self.pools_x[i](x)), self.convs_y[i](self.pools_y[i](y)))
            else:
                x_, y_ = self.attns[i](self.convs_x[i](self.pools_x[i](x)+x_up), self.convs_y[i](self.pools_y[i](y)+y_up))
            res_x = torch.add(res_x, F.interpolate(x_, x_size[2:], mode='bilinear', align_corners=True))
            res_y = torch.add(res_y, F.interpolate(y_, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                x_up = F.interpolate(x_, scale_factor=2, mode='bilinear', align_corners=True)
                y_up = F.interpolate(y_, scale_factor=2, mode='bilinear', align_corners=True)
        res_x = x + self.sum_x(self.relu(res_x))
        res_y = y + self.sum_y(self.relu(res_y))

        return res_x, res_y
    
##########################################################################
class DSAttention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(DSAttention, self).__init__()
        
        self.sizes = [3,5,7]
        self.num_groups = 4
        self.groups = [2,1,1]
        self.sizes_num = len(self.sizes)
        g_dim = dim//self.num_groups
        self.chans_splits = [g_dim*n for n in self.groups]
        HighPassFilters1, HighPassFilters2 = [],[]
        for i,n in zip(self.sizes,self.groups):
            HighPassFilters1.append(nn.Conv2d(g_dim*n, g_dim*n, kernel_size=i, stride=1, padding=i//2, bias=bias, groups=g_dim*n))
            HighPassFilters2.append(nn.Conv2d(g_dim*n, g_dim*n, kernel_size=i, stride=1, padding=i//2, bias=bias, groups=g_dim*n))
        self.HighPassFilters1 = nn.ModuleList(HighPassFilters1)
        self.HighPassFilters2 = nn.ModuleList(HighPassFilters2)
        self.q_h = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias))
        self.k_h = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias))
        self.v_h = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias))
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.attn_blk = nn.Sequential(nn.Conv2d(dim, dim//4, kernel_size=1, bias=bias),nn.ReLU(),nn.Conv2d(dim//4, dim, kernel_size=1, bias=bias),nn.Tanh())

        self.num_head = num_head
        self.LowPassFilters1=nn.AvgPool2d(kernel_size=2, stride=2)
        self.LowPassFilters2=nn.AvgPool2d(kernel_size=2, stride=2)
        self.q_l = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias))
        self.k_l = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias))
        self.v_l = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias))
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        xs = torch.split(x, self.chans_splits, dim=1)
        x_h = [self.HighPassFilters1[i](xs[i]) for i in range(self.sizes_num)]
        x_h = torch.cat(x_h, dim=1)
        ys = torch.split(y, self.chans_splits, dim=1)
        y_h = [self.HighPassFilters2[i](ys[i]) for i in range(self.sizes_num)]
        y_h = torch.cat(y_h, dim=1)
        q_h = self.q_h(x_h)
        k_h = self.k_h(y_h)
        v_h = self.v_h(x_h)  
        high_att = self.attn_blk(self.ap(q_h * k_h))
        out_h = high_att * v_h

        q_l = self.q_l(self.LowPassFilters1(x))
        k_l = self.k_l(self.LowPassFilters2(y))
        v_l = self.v_l(y)
        k_l = rearrange(k_l, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        q_l = rearrange(q_l, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v_l = rearrange(v_l, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        q_l = torch.nn.functional.normalize(q_l, dim=-1)
        k_l = torch.nn.functional.normalize(k_l, dim=-1)
        
        low_att = q_l @ k_l.transpose(-2, -1) * self.temperature
        low_att = self.softmax(low_att)
        out = low_att @ v_l
        out_l = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out_h+out_l)
        return out

# Dual-Domain Adaptive Fusion Module
class DAFM(nn.Module):
    def __init__(self, dim, num_heads, ffn_expand=4, drop_path=0.1, bias=False):
        super(DAFM, self).__init__()

        self.norm1_x = LayerNorm2d(dim)
        self.norm1_y = LayerNorm2d(dim)
        self.attn = DSAttention(dim, num_heads, bias=bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm2d(dim)
        ffn_expand_dim = int(dim * ffn_expand)
        self.ffn = DFFN(dim, hidden_features=ffn_expand_dim)

    def forward(self, x, y):
        # input (b, c, h, w) return (b, c, h, w)
        # x: vis
        # y: nir
        assert x.shape == y.shape, 'The shape of feature maps from target and guidance branch are not equal!'
        b, c, h, w = x.shape
        inp = x
        x = self.norm1_x(x)
        y = self.norm1_y(y)
        x = self.attn(x, y)
        mid = inp + self.drop_path(x)
        # FFN
        #mid = to_3d(mid)
        x = mid + self.drop_path(self.ffn(self.norm2(mid)))
        #x = to_4d(x, h, w)

        return x

##########################################################################
## U-Net
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias, scale_unetfeats, csff, cross=False, num_heads=[1,2,4]):
        super(Encoder, self).__init__()

        self.encoder_level1 = ResBlock(n_feat,                     kernel_size, bias=bias, act=act)
        self.encoder_level2 = ResBlock(n_feat+scale_unetfeats,     kernel_size, bias=bias, act=act)
        self.encoder_level3 = ResBlock(n_feat+(scale_unetfeats*2), kernel_size, bias=bias, act=act)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        if cross:
            self.image_event_transformer1 = DAFM(n_feat, num_heads=num_heads[0], ffn_expand=4, bias=bias)
            self.image_event_transformer2 = DAFM(n_feat+scale_unetfeats, num_heads=num_heads[1], ffn_expand=4, bias=bias)
            self.image_event_transformer3 = DAFM(n_feat+(scale_unetfeats*2), num_heads=num_heads[2], ffn_expand=4, bias=bias)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None, guids=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])
        if guids is not None:
            guid1, guid2, guid3 = guids
            enc1 = self.image_event_transformer1(enc1, guid1)

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])
        if guids is not None:
            enc2 = self.image_event_transformer2(enc2, guid2)

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
        if guids is not None:
            enc3 = self.image_event_transformer3(enc3, guid3)
        
        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = ResBlock(n_feat,                     kernel_size, bias=bias, act=act)
        self.decoder_level2 = ResBlock(n_feat+scale_unetfeats,     kernel_size, bias=bias, act=act)
        self.decoder_level3 = ResBlock(n_feat+(scale_unetfeats*2), kernel_size, bias=bias, act=act)

        self.skip_attn1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.skip_attn2 = conv(n_feat+scale_unetfeats, n_feat+scale_unetfeats, kernel_size, bias=bias)
        
        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]


##########################################################################
class RFFNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, guid_c=1, n_feat=64, scale_unetfeats=32, kernel_size=3, bias=False):
        super(RFFNet, self).__init__()

        act=nn.PReLU()
        self.shallow_feat1 = conv(in_c, n_feat, kernel_size, bias=bias)
        self.shallow_feat2 = conv(in_c, n_feat, kernel_size, bias=bias)
        self.shallow_feat_guid = conv(guid_c, n_feat, kernel_size, bias=bias)

        self.calibrate = MICM(n_feat, bias=bias)

        self.stage1_encoder = Encoder(n_feat, kernel_size, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, act, bias, scale_unetfeats)

        self.guid_encoder = Encoder(n_feat, kernel_size, act, bias, scale_unetfeats, csff=False)

        self.stage2_encoder = Encoder(n_feat, kernel_size, act, bias, scale_unetfeats, csff=True, cross=True, num_heads=[1,2,4])
        self.stage2_decoder = Decoder(n_feat, kernel_size, act, bias, scale_unetfeats)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        
        self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.tail     = conv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, x, guid):

        inp = x
        ##-------------- Stage 1---------------------
        x1 = self.shallow_feat1(inp)
        feat1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat1)

        ## Apply Supervised Attention Module (SAM)
        x2_samfeats, stage1_img = self.sam12(res1[0], inp)
        
        ##-------------- Stage 2---------------------
        x2  = self.shallow_feat2(inp)
        guid  = self.shallow_feat_guid(guid)
        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2_cat = self.concat12(torch.cat([x2, x2_samfeats], 1))

        ## Calibrate guidance features
        x2_cat, guid = self.calibrate(x2_cat, guid)

        ## Fusion
        guids = self.guid_encoder(guid)
        feat2 = self.stage2_encoder(x2_cat, feat1, res1, guids)
        res2 = self.stage2_decoder(feat2)

        stage2_img = self.tail(res2[0])

        return stage2_img+inp, stage1_img
    
if __name__ == "__main__":
    import torch
    from thop import profile

    # Model
    print('==> Building model..')
    model = RFFNet().cuda(0)

    noisy = torch.randn(1, 3, 128, 128).cuda(0)
    guid = torch.randn(1, 1, 128, 128).cuda(0)

    num = 500

    import time
    for i in range(500):
        model(noisy, guid)
    start = time.time()
    for i in range(num):
        torch.cuda.synchronize()
        model(noisy, guid)
        torch.cuda.synchronize()
    time_used = time.time() - start

    flops, params = profile(model, (noisy, guid))
    # print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M, time: %.2f ms' % (flops / 1e9, params / 1e6, time_used / num * 1e3))