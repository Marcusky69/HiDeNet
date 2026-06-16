import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, act=True):
        super(ConvAct, self).__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, 0, dilation=dilation),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            ConvAct(channels, channels, 3, 1),
            ConvAct(channels, channels, 3, 1, act=False),
        )

    def forward(self, x):
        return x + self.body(x)


class MultiScaleBlock(nn.Module):
    """Multi-branch feature extractor for texture and illumination context."""

    def __init__(self, channels):
        super(MultiScaleBlock, self).__init__()
        self.branch_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 0, 0)),
            nn.Conv2d(channels, channels, (1, 3), 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((0, 0, 1, 1)),
            nn.Conv2d(channels, channels, (3, 1), 1, 0),
            nn.ReLU(inplace=True),
        )
        self.branch_5 = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.Conv2d(channels, channels, (1, 5), 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((0, 0, 2, 2)),
            nn.Conv2d(channels, channels, (5, 1), 1, 0),
            nn.ReLU(inplace=True),
        )
        self.branch_dilated = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(channels, channels, 3, 1, 0, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        feat = torch.cat(
            [self.branch_3(x), self.branch_5(x), self.branch_dilated(x)],
            dim=1,
        )
        return x + self.fuse(feat)


class HierarchicalFusion(nn.Module):
    """Fuse shallow texture features with deeper contextual features."""

    def __init__(self, shallow_ch, deep_ch, out_ch):
        super(HierarchicalFusion, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(shallow_ch + deep_ch, out_ch, 1, 1, 0),
            nn.ReLU(inplace=True),
            ConvAct(out_ch, out_ch, 3, 1),
        )

    def forward(self, shallow, deep):
        deep = F.interpolate(
            deep,
            size=shallow.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fuse(torch.cat([shallow, deep], dim=1))


class N_net(nn.Module):
    """Proxy image estimator: lightweight residual CNN without downsampling."""

    def __init__(self, num=32, blocks=3):
        super(N_net, self).__init__()
        self.head = ConvAct(3, num, 3, 1)
        self.body = nn.Sequential(*[ResidualBlock(num) for _ in range(blocks)])
        self.tail = nn.Sequential(
            ConvAct(num, num, 3, 1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )

    def forward(self, input):
        feat = self.body(self.head(input))
        residual = self.tail(feat)
        return torch.sigmoid(input + residual)


class L_net(nn.Module):
    """Illumination predictor. Outputs a single-channel smooth illumination map."""

    def __init__(self, num=64):
        super(L_net, self).__init__()
        self.shallow = nn.Sequential(
            ConvAct(3, num, 3, 1),
            MultiScaleBlock(num),
        )
        self.down = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num * 2, 3, 2, 0),
            nn.ReLU(inplace=True),
            MultiScaleBlock(num * 2),
            MultiScaleBlock(num * 2),
        )
        self.fusion = HierarchicalFusion(num, num * 2, num)
        self.out = nn.Sequential(
            ConvAct(num, num // 2, 3, 1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num // 2, 1, 3, 1, 0),
        )

    def forward(self, input):
        shallow = self.shallow(input)
        deep = self.down(shallow)
        feat = self.fusion(shallow, deep)
        return torch.sigmoid(self.out(feat))


class R_net(nn.Module):
    """Reflectance predictor. Preserves texture/color with hierarchical fusion."""

    def __init__(self, num=64):
        super(R_net, self).__init__()
        self.shallow = nn.Sequential(
            ConvAct(3, num, 3, 1),
            MultiScaleBlock(num),
        )
        self.down = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num * 2, 3, 2, 0),
            nn.ReLU(inplace=True),
            MultiScaleBlock(num * 2),
            MultiScaleBlock(num * 2),
        )
        self.fusion = HierarchicalFusion(num, num * 2, num)
        self.refine = nn.Sequential(
            MultiScaleBlock(num),
            ConvAct(num, num, 3, 1),
        )
        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 3, 3, 1, 0),
        )

    def forward(self, input):
        shallow = self.shallow(input)
        deep = self.down(shallow)
        feat = self.refine(self.fusion(shallow, deep))
        return torch.sigmoid(self.out(feat))


class HiDeNet(nn.Module):
    """Proxy-conditioned Retinex decomposition network."""

    def __init__(self, num=64, proxy_num=32, gamma=0.6):
        super(HiDeNet, self).__init__()
        self.gamma = gamma
        self.N_net = N_net(num=proxy_num)
        self.L_net = L_net(num=num)
        self.R_net = R_net(num=num)

    def enhance(self, reflectance, illumination, gamma=None):
        gamma = self.gamma if gamma is None else gamma
        enhanced = reflectance * torch.pow(illumination.clamp_min(1e-4), gamma)
        return enhanced.clamp(0.0, 1.0)

    def forward(self, input, gamma=None, return_dict=False):
        proxy = self.N_net(input)
        illumination = self.L_net(proxy)
        reflectance = self.R_net(proxy)
        enhanced = self.enhance(reflectance, illumination, gamma=gamma)

        if return_dict:
            return {
                "enhanced": enhanced,
                "illumination": illumination,
                "reflectance": reflectance,
                "proxy": proxy,
            }
        return illumination, reflectance, proxy, enhanced


class net(HiDeNet):
    """Backward-compatible wrapper for existing imports."""

    def __init__(self):
        super(net, self).__init__(num=64, proxy_num=32, gamma=0.6)

    def forward(self, input):
        output = super(net, self).forward(input, return_dict=True)
        return output["illumination"], output["reflectance"], output["proxy"]


def make_photometric_views(x, strength=0.15):
    """Create two differentiable photometric views from one low-light batch."""

    if strength <= 0:
        return x, x

    b, c, _, _ = x.shape
    device = x.device
    dtype = x.dtype

    gain1 = 1.0 + (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * strength
    gain2 = 1.0 + (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * strength
    bias1 = (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * strength * 0.1
    bias2 = (torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * strength * 0.1
    color1 = 1.0 + (torch.rand(b, c, 1, 1, device=device, dtype=dtype) * 2 - 1) * strength * 0.5
    color2 = 1.0 + (torch.rand(b, c, 1, 1, device=device, dtype=dtype) * 2 - 1) * strength * 0.5

    view1 = (x * gain1 * color1 + bias1).clamp(0.0, 1.0)
    view2 = (x * gain2 * color2 + bias2).clamp(0.0, 1.0)
    return view1, view2


def gradient_x(x):
    return x[:, :, :, 1:] - x[:, :, :, :-1]


def gradient_y(x):
    return x[:, :, 1:, :] - x[:, :, :-1, :]


def illumination_smoothness_loss(illumination, image):
    image_gray = image.mean(dim=1, keepdim=True)
    weight_x = torch.exp(-10.0 * torch.mean(torch.abs(gradient_x(image_gray)), dim=1, keepdim=True))
    weight_y = torch.exp(-10.0 * torch.mean(torch.abs(gradient_y(image_gray)), dim=1, keepdim=True))
    loss_x = torch.abs(gradient_x(illumination)) * weight_x
    loss_y = torch.abs(gradient_y(illumination)) * weight_y
    return loss_x.mean() + loss_y.mean()


class PairedLowLightSelfSupervisedLoss(nn.Module):
    """Paired-view loss with reflectance consistency for low-light training."""

    def __init__(
        self,
        lambda_recon=1.0,
        lambda_reflectance=0.2,
        lambda_smooth=0.1,
        view_strength=0.15,
    ):
        super(PairedLowLightSelfSupervisedLoss, self).__init__()
        self.lambda_recon = lambda_recon
        self.lambda_reflectance = lambda_reflectance
        self.lambda_smooth = lambda_smooth
        self.view_strength = view_strength

    def forward(self, model, low_light):
        view1, view2 = make_photometric_views(low_light, self.view_strength)
        out1 = model(view1, return_dict=True)
        out2 = model(view2, return_dict=True)

        recon1 = F.l1_loss(out1["reflectance"] * out1["illumination"], view1)
        recon2 = F.l1_loss(out2["reflectance"] * out2["illumination"], view2)
        reflectance = F.l1_loss(out1["reflectance"], out2["reflectance"].detach())
        reflectance = reflectance + F.l1_loss(out2["reflectance"], out1["reflectance"].detach())
        smooth = illumination_smoothness_loss(out1["illumination"], view1)
        smooth = smooth + illumination_smoothness_loss(out2["illumination"], view2)

        total = (
            self.lambda_recon * (recon1 + recon2)
            + self.lambda_reflectance * reflectance
            + self.lambda_smooth * smooth
        )
        return {
            "loss": total,
            "reconstruction": recon1 + recon2,
            "reflectance_consistency": reflectance,
            "illumination_smoothness": smooth,
            "view1": out1,
            "view2": out2,
        }
