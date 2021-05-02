import torch
import torch.nn as nn
import numpy as np
from models.entransformer import Block, Mlp, Attention
from models.patchembed import PatchEmbed
from models.fuse_layer import FuseLayer
import yaml
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class HRTransModel(nn.Module):
    def __init__(self, config, num_branches, block, dpr, stage_index, norm_layer):
        super(HRTransModel, self).__init__()
        # self.config = config
        self.num_branches = num_branches
        self.config = config['MODULE']['LAYERS']
        self.st_config = config['MODULE']['STAGES']
        self.layer1_cfig = self.config['LAYER1']
        self.layer2_cfig = self.config['LAYER4']
        self.layer3_cfig = self.config['LAYER3']
        self.layer4_cfig = self.config['LAYER2']

        self.stage1_cfig = self.st_config['STAGE1']
        self.stage2_cfig = self.st_config['STAGE2']
        self.stage3_cfig = self.st_config['STAGE3']
        self.stage4_cfig = self.st_config['STAGE4']

        self.num_fuse = self.st_config['NUM_FUSE'][stage_index]
        self.la_configs = [self.layer1_cfig, self.layer2_cfig, self.layer3_cfig, self.layer4_cfig]
        self.st_configs = [self.stage1_cfig, self.stage2_cfig, self.stage3_cfig, self.stage4_cfig]

        self.branches = self._make_branches(self.la_configs, block=block, norm_layer=norm_layer, dpr=dpr)

        self.stage_index = stage_index

        self.fuse_layer = self._make_fuse_layer()

        self.relu = nn.ReLU()

    # TODO 对于下采样及上采样进行融合，考虑有两种方法，一种是恢复成欧几里得空间的额图，然后使用卷积操作，另一种是使用Patchembed方法，使用全连接层进行维度转换
    def _make_fuse_layer(self):
        # assert len(x) == in_nums, f"input num {len(x)} should be the same as check_num {in_nums}"
        fuse_config = self.st_config[self.stage_index]
        in_nums = fuse_config['LA_NUMS']
        in_channels = fuse_config['IN_CHANNELS']
        out_nums = in_nums or 1
        out_channels = in_channels or 1
        assert in_nums == len(
            in_channels) and out_nums == out_channels, f"should in_nums {in_nums} == len(in_channels) " \
                                                       f"{len(in_channels)} and out_nums {out_nums} " \
                                                       f"== out_channels {len(out_channels)}"
        fuse_layers = []
        for i in range(in_nums):
            fuse_layer = []
            for j in range(out_nums):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels[j], out_channels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(out_channels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest")
                        )
                    )
                elif (j == i):
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = in_channels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )

                        else:
                            num_outchannels_conv3x3 = in_channels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
                fuse_layers.append(nn.ModuleList(fuse_layer))
            return nn.ModuleList(fuse_layers)

    def _make_fuse_stage_l(self, x, in_num, out_num):
        fus_layers = []
        pass

    def _make_one_branch(config, block, norm_layer, dpr, cur):
        layers = []
        # num_blocks；每个全连接之间有几个对应的trasformer块
        print(config['NUM_BLOCKS'])
        for i in config['NUM_BLOCKS']:
            print("i", i)
            layer = [block(
                dim=config['EMBEDED_DIM'], num_heads=config['NUM_HEADS'], mlp_ratio=config['MLP_RATIO'],
                qkv_bias=config['QKV_BIAS'], qk_scale=config['QK_SCALE'], proj_drop=config['PROJ_DROP']
                , attn_drop=config['ATTN_DROP'], drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=config['SR_RATIO'])
                for j in range(i)
            ]
            layers.append(nn.Sequential(*layer))
            print("len", len(layers))
        return nn.ModuleList(layers)

    def _make_branches(self, configs, block, norm_layer, dpr):
        branches = []
        cur = 0
        print("num", self.num_branches)
        for i in range(self.num_branches):
            branches.append(self._make_one_branch(configs[i], block, norm_layer, dpr, cur))
            cur += 1
        return nn.ModuleList(branches)

    # x 是个列表，表示上一阶段的所有输出
    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0][0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i][self.stage_index][x[i]]

        x_fuse = []

        for i in range(self.num_fuse):
            y = x[0] if i == 0 else self.branches[0][self.stage_index](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[i]
                else:
                    y = y + self.fuse_layer[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class PVT_Feature(nn.Module):
    def __init__(self, config, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.):
        super(PVT_Feature, self).__init__()
        # self.config = config
        self.config = config['PVT-FEATURES']
        self.embed_dims = self.config['EMBEDED_DIMS']
        # 使用多少个transformer
        self.num_blocks = self.config['NUM_BLOCKS']
        # 不同层中的每个transformer的多头自注意力中头的数目
        self.num_heads = self.config['NUM_HEADS']
        # mlp中的隐藏层的结点数是输入结点的多少倍，如果不指定输出的结点数目，则默认输出与输入相同
        self.mlp_ratio = self.config['MLP_RATIO']
        # 在自注意力的最后映射中的有多少的结点被失活
        self.proj_drop = self.config['PROJ_DROP']
        self.sr_ratios = self.config['SR_RATIOS']
        self.patchembed1 = PatchEmbed(img_size=256, patch_size=2, in_channels=3, embed_dim=self.embed_dims[0])
        self.patchembed2 = PatchEmbed(img_size=128, patch_size=2, in_channels=self.embed_dims[0],
                                      embed_dim=self.embed_dims[1])
        self.patchembed3 = PatchEmbed(img_size=64, patch_size=2, in_channels=self.embed_dims[1],
                                      embed_dim=self.embed_dims[2])
        self.patchembed4 = PatchEmbed(img_size=32, patch_size=2, in_channels=self.embed_dims[2],
                                      embed_dim=self.embed_dims[3])
        self.patchembed = [self.patchembed1, self.patchembed2, self.patchembed3, self.patchembed4]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.num_blocks))]  # stochastic depth decay rule
        # block1 用于对输入的初始图片进行预处理，所以此处的Block块不一个，具体多少个，在yaml文件中配置

        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=self.embed_dims[0], num_heads=self.num_heads[0], mlp_ratio=self.mlp_ratio[0], qkv_bias=True,
            qk_scale=None,
            proj_drop=self.proj_drop[0], attn_drop=0., drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=self.sr_ratios[0])
            for i in range(self.num_blocks[0])
        ])
        self.block2 = nn.ModuleList([Block(
            dim=self.embed_dims[1], num_heads=self.num_heads[1], mlp_ratio=self.mlp_ratio[1], qkv_bias=True,
            qk_scale=None,
            proj_drop=self.proj_drop[1], attn_drop=0., drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=self.sr_ratios[1])
            for i in range(self.num_blocks[1])
        ])
        self.block3 = nn.ModuleList([Block(
            dim=self.embed_dims[2], num_heads=self.num_heads[2], mlp_ratio=self.mlp_ratio[2], qkv_bias=True,
            qk_scale=None,
            proj_drop=self.proj_drop[2], attn_drop=0., drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=self.sr_ratios[2])
            for i in range(self.num_blocks[2])
        ])
        self.block4 = nn.ModuleList([Block(
            dim=self.embed_dims[3], num_heads=self.num_heads[3], mlp_ratio=self.mlp_ratio[3], qkv_bias=True,
            qk_scale=None,
            proj_drop=self.proj_drop[3], attn_drop=0., drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=self.sr_ratios[3])
            for i in range(self.num_blocks[3])
        ])
        self.block = [self.block1, self.block2, self.block3, self.block4]

        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patchembed1.num_patches, self.embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patchembed2.num_patches, self.embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patchembed3.num_patches, self.embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patchembed4.num_patches, self.embed_dims[3]))
        self.pos_drop4 = nn.Dropout(p=drop_rate)
        self.pos_embed = [self.pos_embed1, self.pos_embed2, self.pos_embed3, self.pos_embed4]
        self.pos_drop = [self.pos_drop1, self.pos_drop2, self.pos_drop3, self.pos_drop4]

        self.norm1 = nn.LayerNorm(self.embed_dims[0])
        self.norm2 = nn.LayerNorm(self.embed_dims[1])
        self.norm3 = nn.LayerNorm(self.embed_dims[2])
        self.norm4 = nn.LayerNorm(self.embed_dims[3])
        self.norm = [self.norm1, self.norm2, self.norm3, self.norm4]

        # self.conv8_1 = nn.Conv2d(8,1,1,1,0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patchembed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward(self, x, index):

        x, (H, W) = self.patchembed[index](x)
        pos_embed = self._get_pos_embed(self.pos_embed[index], self.patchembed[index], H, W)
        x = x + pos_embed
        x = self.pos_drop[index](x)
        for blk in self.block[index]:
            x = blk(x, H, W)

        return x, (H, W)


# TODO HRNet 的模型有两种，一种是边缘完全靠PVT，二是边缘的输出靠上层输出
class HRTransNet(nn.Module):
    def __init__(self, config):
        super(HRTransNet, self).__init__()
        self.pvt_feature = PVT_Feature(config)
        self.norm_layer = nn.LayerNorm
        self.block = Block
        self.relu = nn.ReLU(inplace=True)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn16 = nn.BatchNorm2d(16)

        self.config = config['MODULE']
        self.pvt_config = config['PVT-FEATURES']
        self.stage1 = self.config['STAGES']['STAGE1']
        self.stage2 = self.config['STAGES']['STAGE2']
        self.stage3 = self.config['STAGES']['STAGE3']
        self.stage4 = self.config['STAGES']['STAGE4']

        self.st1_inchannels = self.stage1['IN_CHANNELS']
        self.st2_inchannels = self.stage2['IN_CHANNELS']
        self.st3_inchannels = self.stage3['IN_CHANNELS']
        self.st4_inchannels = self.stage4['IN_CHANNELS']

        self.layer1_cfg = self.config['LAYERS']['LAYER1']
        self.layer2_cfg = self.config['LAYERS']['LAYER2']
        self.layer3_cfg = self.config['LAYERS']['LAYER3']
        self.layer4_cfg = self.config['LAYERS']['LAYER4']

        self.la1_blocks = self.layer1_cfg['NUM_BLOCKS']
        self.la2_blocks = self.layer2_cfg['NUM_BLOCKS']
        self.la3_blocks = self.layer3_cfg['NUM_BLOCKS']
        self.la4_blocks = self.layer4_cfg['NUM_BLOCKS']
        self.image_size = self.config['IMAGE_SIZE']

        self.layers_config = [self.layer1_cfg, self.layer2_cfg, self.layer3_cfg, self.layer4_cfg]
        self.num_fuses = self.config['LAYERS']['NUM_FUSES']

        # FuseLayer参数说明: 2:表示有几个层进行全连接
        #                  [32, 64]:长度与 2 对应，表示每个层输入进来的通道数
        #                   2:此次全连接融合的个数
        self.make_fuse_layer2 = FuseLayer(2, [32, 64], 2)
        self.make_fuse_layer3 = FuseLayer(3, [32, 64, 128], 2)
        self.make_fuse_layer4 = FuseLayer(4, [32, 64, 128, 256], 2)

        # self.make_fuse_layer3 = self._make_fuse_layer(3, self.st3_inchannels)
        # self.make_fuse_layer4 = self._make_fuse_layer(4, self.st4_inchannels)
        # drop_path_rate
        self.dpr = [x.item() for x in torch.linspace(0, 0.1, sum(self.la1_blocks) * 4)]
        # 分支处理模块
        self.branchs_module = self._make_branch(self.layers_config, self.block, self.norm_layer, self.dpr, 4)

        # assert self.num_fuses == len(self.la1_blocks)-1,f"Please check config,self.num_fuses {self.num_fuses} should agree with " \
        #                                                 f"len(self.la1_blocks)-1 {len(self.la1_blocks)-1}"

        self.keep_embed1_1 = PatchEmbed(img_size=self.layer1_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][0],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][0])
        self.keep_embed1_2 = PatchEmbed(img_size=self.layer1_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][0],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][0])
        self.keep_embed1_3 = PatchEmbed(img_size=self.layer1_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][0],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][0])
        self.keep_embed1_4 = PatchEmbed(img_size=self.layer1_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][0],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][0])
        self.keep_embed2_1 = PatchEmbed(img_size=self.layer2_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][1],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][1])
        self.keep_embed2_2 = PatchEmbed(img_size=self.layer2_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][1],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][1])
        self.keep_embed2_3 = PatchEmbed(img_size=self.layer2_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][1],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][1])
        self.keep_embed3_1 = PatchEmbed(img_size=self.layer3_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][2],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][2])
        self.keep_embed3_2 = PatchEmbed(img_size=self.layer3_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][2],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][2])
        self.keep_embed3_3 = PatchEmbed(img_size=self.layer3_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][2],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][2])
        self.keep_embed4_1 = PatchEmbed(img_size=self.layer4_cfg['IMG_SIZE'], patch_size=1,
                                        in_channels=self.pvt_config['EMBEDED_DIMS'][3],
                                        embed_dim=self.pvt_config['EMBEDED_DIMS'][3])

        # self.k_pose_embed1_1 = nn.Parameter(
        #     torch.zeros(1, self.keep_embed1_1.num_patches, self.pvt_config['EMBEDED_DIMS'][0]))
        # self.pos_drop1_1 = nn.Dropout(p=0.1)
        # self.k_pose_embed1_2 = nn.Parameter(
        #     torch.zeros(1, self.keep_embed1_2.num_patches, self.pvt_config['EMBEDED_DIMS'][0]))
        # self.pos_drop1_2 = nn.Dropout(p=0.1)
        # self.k_pose_embed1_3 = nn.Parameter(
        #     torch.zeros(1, self.keep_embed1_3.num_patches, self.pvt_config['EMBEDED_DIMS'][0]))
        # self.pos_drop1_3 = nn.Dropout(p=0.1)
        # self.k_pose_embed1_4 = nn.Parameter(
        #     torch.zeros(1, self.keep_embed1_4.num_patches, self.pvt_config['EMBEDED_DIMS'][0]))
        # self.pos_drop1_4 = nn.Dropout(p=0.1)
        #
        # self.k_pose_embed2_1 = nn.Parameter(
        #     torch.zeros(1, self.keep_embed2_1.num_patches, self.pvt_config['EMBEDED_DIMS'][1]))
        # self.pos_drop2_1 = nn.Dropout(p=0.1)
        # self.k_pose_embed2_2 = nn.Parameter(
        #     torch.zeros(1, self.keep_embed2_2.num_patches, self.pvt_config['EMBEDED_DIMS'][1]))
        # self.pos_drop2_2 = nn.Dropout(p=0.1)
        #
        # self.k_pose_embed3_1 = nn.Parameter(
        #     torch.zeros(1, self.keep_embed3_1.num_patches, self.pvt_config['EMBEDED_DIMS'][2]))
        # self.pos_drop3_1 = nn.Dropout(p=0.1)
        # self.k_pose_embed3_2 = nn.Parameter(
        #     torch.zeros(1, self.keep_embed3_2.num_patches, self.pvt_config['EMBEDED_DIMS'][2]))
        # self.pos_drop2_2 = nn.Dropout(p=0.1)

        self.sal_st1_up = nn.Sequential(
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM'], self.layer1_cfg['EMBEDED_DIM']//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM']//2, self.layer1_cfg['EMBEDED_DIM']//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM']//2, 1, 3, 1, 1)
        )
        self.sal_st2_up = nn.Sequential(
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM'], self.layer1_cfg['EMBEDED_DIM'] // 2, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM'] // 2, self.layer1_cfg['EMBEDED_DIM'] // 2, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM'] // 2, 1, 3, 1, 1)
        )
        self.sal_st3_up = nn.Sequential(
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM'], self.layer1_cfg['EMBEDED_DIM'] // 2, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM'] // 2, self.layer1_cfg['EMBEDED_DIM'] // 2, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM'] // 2, 1, 3, 1, 1)
        )

        self.fin_score = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM'], self.layer1_cfg['EMBEDED_DIM'] // 2, 3, 1, 1),
            nn.BatchNorm2d(self.layer1_cfg['EMBEDED_DIM'] // 2),
            nn.ReLU(),
            nn.Conv2d(self.layer1_cfg['EMBEDED_DIM'] // 2, 1, 1, 1, 0)
        )

    def _make_one_branch(self, config, block, norm_layer, dpr):
        layers = []
        block_layers = []
        # num_blocks；每个全连接之间有几个对应的trasformer块
        cur = 0
        # print(config['EMBEDED_DIM'])
        for i in config['NUM_BLOCKS']:
            for k in range(config['NUM_BIG_BLOCKS']):
                layer = [block(
                    dim=config['EMBEDED_DIM'], num_heads=config['NUM_HEADS'], mlp_ratio=config['MLP_RATIO'],
                    qkv_bias=config['QKV_BIAS'], qk_scale=config['QK_SCALE'], proj_drop=config['PROJ_DROP']
                    , attn_drop=config['ATTN_DROP'], drop_path=dpr[cur + j], norm_layer=norm_layer,
                    sr_ratio=config['SR_RATIO'])
                    for j in range(i)
                ]
                cur += i
                layers.append(nn.Sequential(*layer))
            block_layers.append(nn.ModuleList(layers))
        return nn.ModuleList(block_layers)

    def _make_branch(self, config, block, norm_layer, dpr, num_branches):
        branch_layer = []
        for i in range(num_branches):
            branch_layer.append(self._make_one_branch(config[i], block, norm_layer, dpr))

        return nn.ModuleList(branch_layer)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.keep_embed1_1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward(self, x):
        # x0 最开始经过pvt处理得到的特征
        B = x.shape[0]
        features = []
        # stage1
        x0_1, (H0, W0) = self.pvt_feature(x, 0)  #

        for i in range(3):
            x0 = self.branchs_module[0][0][0][i](x0_1, H0, W0)
        x0_t = x0.reshape(B, H0, W0, -1).permute(0, 3, 1, 2).contiguous()

        sal_st1 = self.sal_st1_up(x0_t)
        features.append(F.interpolate(sal_st1, size=self.image_size, mode='bilinear', align_corners=True))

        x1, (H1, W1) = self.pvt_feature(x0_t, 1)  # 获得第二层的输入特征
        for i in range(3):
            for j in range(3):
                x0 = self.branchs_module[0][1][i][j](x0, H0, W0)
                x1 = self.branchs_module[1][0][i][j](x1, H1, W1)

        x0_t = x0.reshape(B, H0, W0, -1).permute(0, 3, 1, 2).contiguous()
        x1_t = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        in_fuse2 = [x0_t, x1_t]
        out_fuse2 = self.make_fuse_layer2(in_fuse2)

        sal_st1 = self.sal_st2_up(out_fuse2[0])
        features.append(F.interpolate(sal_st1, self.image_size,mode='bilinear', align_corners=True))

        x0_3, (H0, W0) = self.keep_embed1_2(out_fuse2[0])
        x1_3, (H1, W1) = self.keep_embed2_2(out_fuse2[1])
        x2_3, (H2, W2) = self.pvt_feature(out_fuse2[1], 2)

        for i in range(3):
            for j in range(3):
                x0_3 = self.branchs_module[0][2][i][j](x0_3, H0, W0)
                x1_3 = self.branchs_module[1][1][i][j](x1_3, H1, W1)
                x2_3 = self.branchs_module[2][0][i][j](x2_3, H2, W2)
        x0_3_t = x0_3.reshape(B, H0, W0, -1).permute(0, 3, 1, 2).contiguous()
        x1_3_t = x1_3.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2_3_t = x2_3.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        in_fuse3 = [x0_3_t, x1_3_t, x2_3_t]
        out_fuse3 = self.make_fuse_layer3(in_fuse3)
        sal_st3 = self.sal_st3_up(out_fuse3[0])
        features.append(F.interpolate(sal_st3, size=self.image_size, mode='bilinear', align_corners=True))

        fin_out = self.fin_score(out_fuse3[0])
        features.append(fin_out)

        return features


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


if __name__ == "__main__":
    a = np.random.random((1, 3, 256, 256))
    b = torch.Tensor(a)
    path = '../config/module.yaml'
    config = yaml.load(open(path, 'r'), yaml.SafeLoader)

    # lf = PVT_Feature(config)
    hrtransnet = HRTransNet(config)
    hrtransnet(b)

    # out = lf(b)
    # for f in out:
    #     print(f.shape)
