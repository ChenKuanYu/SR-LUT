
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys


# USER PARAMS
UPSCALE = 2                  # upscaling factor
#MODEL_PATH = "./Model_S.pth"    # Trained SR net params
MODEL_PATH = "./model_G_i200000.pth"    # Trained SR net params
SAMPLING_INTERVAL = 4        # N bit uniform sampling


class RC_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), filter_num=64, bias=True):
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1, "kernel_size must be odd number"
        super(RC_Module, self).__init__()
        self.kernel_size = kernel_size
        pad_vertical, pad_horizontal = (kernel_size[0])//2, (kernel_size[1])//2
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=(pad_vertical, pad_horizontal))

        self.linear_in = nn.ModuleList([nn.Linear(in_channels, filter_num) for _ in range(kernel_size[0] * kernel_size[1])])
        self.linear_out = nn.ModuleList([nn.Linear(filter_num, out_channels) for _ in range(kernel_size[0] * kernel_size[1])])

    def forward(self, x):
        # Reshape the x to make it easy to manipulate
        x_shape = x.shape
        x = self.unfold(x)  # B, C*k*k, L
        x = x.reshape(x_shape[0], x_shape[1], self.kernel_size[0] * self.kernel_size[1], -1)  # B, C, k*k, L
        x = x.permute(0, 3, 1, 2)  # B, L, C, k*k
        x = x.reshape(-1, x_shape[1], self.kernel_size[0] * self.kernel_size[1])  # B*L, C, k*k

        # Linear transform the input
        x_kv = [linear_in(x[:, :, i]) for i, linear_in in enumerate(self.linear_in)]
        x_out = [linear_out(x_kv[i]) for i, linear_out in enumerate(self.linear_out)]  # [(B*L, Cout), ...]

        # Average the x_out list to get the output
        x_out = torch.stack(x_out, dim=-1)  # B*L, Cout, k*k
        x_out = x_out.mean(dim=-1)  # B*L, Cout
    
        # Reshape back to B, C, H, W
        Cout_dim = x_out.shape[1]
        x_out = x_out.reshape(x_shape[0], -1, Cout_dim)  # B, L, Cout
        x_out = x_out.permute(0, 2, 1)  # B, Cout, L
        x_out = x_out.reshape(x_shape[0], Cout_dim, x_shape[2], x_shape[3])  # B, Cout, H, W
        return x_out


### A lightweight deep network ###
class Pure_SRNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(Pure_SRNet, self).__init__()

        self.upscale = upscale
        self.conv1 = nn.Conv2d(1, 64, [2,2], stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(64, 1*upscale*upscale, 1, stride=1, padding=0, dilation=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B*C, 1, H, W)
        x = self.conv1(x_in)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale*(H-1), self.upscale*(W-1))

        return x

### A lightweight deep network ###
class SRNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(SRNet, self).__init__()

        self.upscale = upscale
        self.rc_module = RC_Module(in_channels=1, out_channels=1, kernel_size=(5,5), filter_num=64)
        self.conv1 = nn.Conv2d(1, 64, [2,2], stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(64, 1*upscale*upscale, 1, stride=1, padding=0, dilation=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B*C, 1, H, W)
        x_in = self.rc_module(x_in)
        x = self.conv1(x_in)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale*(H-1), self.upscale*(W-1))

        return x

model_G = SRNet(upscale=UPSCALE).cuda()

pure_SRNet = Pure_SRNet(upscale=UPSCALE).cuda()
pure_RCModule = RC_Module(1, 1, (5,5), 64).cuda()

lm = torch.load('{}'.format(MODEL_PATH))
model_G.load_state_dict(lm.state_dict(), strict=True)

#for param in model_G.parameters():
#    print(param.data)
print(model_G)

with torch.no_grad():
    for i in range(1, 7):
        getattr(pure_SRNet, 'conv{}'.format(i)).weight.copy_(getattr(model_G, 'conv{}'.format(i)).weight)
        getattr(pure_SRNet, 'conv{}'.format(i)).bias.copy_(getattr(model_G, 'conv{}'.format(i)).bias)
    pure_SRNet.conv1.weight.copy_(model_G.conv1.weight)
    pure_SRNet.conv1.bias.copy_(model_G.conv1.bias)
    
    for i in range(0, 25):
        pure_RCModule.linear_in[i].weight.copy_(model_G.rc_module.linear_in[i].weight)
        pure_RCModule.linear_in[i].bias.copy_(model_G.rc_module.linear_in[i].bias)

### Extract input-output pairs
with torch.no_grad():
    pure_SRNet.eval()

    # 1D input
    base = torch.arange(0, 257, 2**SAMPLING_INTERVAL)   # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.cuda().repeat(L)                             # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    third = base.cuda().unsqueeze(1).repeat(1, L*L).reshape(-1) # 256*256*256   0 x65536|1 x65536|...|255 x65536
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)    # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L*L*L).reshape(-1) # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)    # [256*256*256*256, 4]

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,1,2,2).float() / 255.0
    print("Input size: ", input_tensor.size())

    # Split input to not over GPU memory
    B = input_tensor.size(0) // 100
    outputs = []

    for b in range(100):
        if b == 99:
            batch_output = pure_SRNet(input_tensor[b*B:])
        else:
            batch_output = pure_SRNet(input_tensor[b*B:(b+1)*B])

        results = torch.round(torch.clamp(batch_output, -1, 1)*127).cpu().data.numpy().astype(np.int8)
        outputs += [ results ]
    
    results = np.concatenate(outputs, 0)
    print("Resulting LUT size: ", results.shape)

    np.save("ex_Model_S_x{}_{}bit_int8".format(UPSCALE, SAMPLING_INTERVAL), results)





