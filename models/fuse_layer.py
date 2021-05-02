import torch
import torch.nn as nn


def _make_fuse_layer(in_nums, in_channels):
    # assert len(x) == in_nums, f"input num {len(x)} should be the same as check_num {in_nums}"
    assert len(in_channels) == in_nums
    out_nums = in_nums or 1
    out_channels = in_channels
    # assert in_nums == len(
    #     in_channels) and out_nums == out_channels, f"should in_nums {in_nums} == len(in_channels) " \
    #                                                f"{len(in_channels)} and out_nums {out_nums} " \
    #                                                f"== out_channels {len(out_channels)}"
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
            elif j == i:
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


class FuseLayer(nn.Module):
    def __init__(self, in_nums, in_channels, fuse_num):
        super(FuseLayer, self).__init__()
        self.make_fuse = _make_fuse_layer(in_nums=in_nums, in_channels=in_channels)
        self.fuse_num = fuse_num
        self.in_nums = in_nums
        self.out_nums = self.in_nums
        self.in_channels = in_channels
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.in_nums == 1:
            return x[0]

        x_fuse = []
        for i in range(self.in_nums):
            y = x[0] if i == 0 else self.make_fuse[i][0](x[0])
            for j in range(1, self.out_nums):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.make_fuse[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse
