class SGBlock(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)  # + 16

        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio

        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                Conv2D(
                    inp, inp, 3, 1, 1, groups=inp, bias_attr=False),
                BatchNorm2D(inp),
                nn.ReLU6(),
                # pw-linear
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                # pw-linear
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                Conv2D(
                    oup, oup, 3, stride, 1, groups=oup, bias_attr=False),
                BatchNorm2D(oup))
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                # pw-linear
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6())
        elif inp != oup and stride == 2 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                # pw-linear
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                Conv2D(
                    oup, oup, 3, stride, 1, groups=oup, bias_attr=False),
                BatchNorm2D(oup))
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                Conv2D(
                    inp, inp, 3, 1, 1, groups=inp, bias_attr=False),
                BatchNorm2D(inp),
                nn.ReLU6(),
                # pw
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                #nn.ReLU6(),
                # pw
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                Conv2D(
                    oup, oup, 3, 1, 1, groups=oup, bias_attr=False),
                BatchNorm2D(oup))

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            if self.identity_div == 1:
                out = out + x
            else:
                shape = x.shape
                id_tensor = x[:, :shape[1] // self.identity_div, :, :]
                out[:, :shape[1] // self.identity_div, :, :] = \
                    out[:, :shape[1] // self.identity_div, :, :] + id_tensor

        return out
