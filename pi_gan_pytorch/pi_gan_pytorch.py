import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helper

def exists(val):
    return val is not None

def leaky_relu(p = 0.2):
    return nn.LeakyReLU(p)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 30., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, gamma = None, beta = None):
        out =  F.linear(x, self.weight, self.bias)

        # FiLM modulation

        if exists(gamma):
            out = out * gamma[:, None, :]

        if exists(beta):
            out = out + beta[:, None, :]

        out = self.activation(out)
        return out

# mapping network

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class MappingNetwork(nn.Module):
    def __init__(self, *, dim, dim_out, depth = 8, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(dim, dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

        self.to_gamma = nn.Linear(dim, dim_out)
        self.to_beta = nn.Linear(dim, dim_out)

    def forward(self, x):
        x = F.normalize(x, dim = 1)
        x = self.net(x)
        return self.to_gamma(x), self.to_beta(x)

# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 30., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, gamma, beta):
        for layer in self.layers:
            x = layer(x, gamma, beta)
        return self.last_layer(x)

# generator

class Generator(nn.Module):
    def __init__(self, image_size, dim, dim_hidden, siren_num_layers = 6):
        super().__init__()

        self.mapping = MappingNetwork(
            dim = dim,
            dim_out = dim_hidden
        )

        self.siren = SirenNet(
            dim_in = 2,
            dim_hidden = dim_hidden,
            dim_out = dim_hidden,
            num_layers = siren_num_layers
        )

        self.to_alpha = nn.Linear(dim_hidden, 1)

        self.to_rgb_siren = Siren(
            dim_in = dim_hidden + 2,
            dim_out = dim_hidden
        )

        self.to_rgb= nn.Linear(dim_hidden, 3)

    def forward(self, latent, ray_direction, coors):
        gamma, beta = self.mapping(latent)
        x = self.siren(coors, gamma, beta)
        alpha = self.to_alpha(x)

        x = torch.cat((x, ray_direction), dim = -1)
        x = self.to_rgb_siren(x, gamma, beta)
        rgb = self.to_rgb(x)
        return rgb, alpha

class Generator(nn.Module):
    def __init__(self, image_size, dim, dim_hidden):
        super().__init__()
        self.image_size = image_size

        coors = torch.stack(torch.meshgrid(
            torch.arange(image_size),
            torch.arange(image_size)
        ))

        coors = rearrange(coors, 'c h w -> (h w) c')
        self.register_buffer('coors', coors)

        self.G = Generator(
            image_size = image_size,
            dim = dim,
            dim_hidden = dim_hidden
        )

    def forward(self, x, ray_direction):
        device, b = x.device, x.shape[0]
        coors = repeat(self.coors, 'n c -> b n c', b = b).float()
        ray_direction = repeat(ray_direction, 'b c -> b n c', n = coors.shape[1])
        return self.G(x, ray_direction, coors)

class DiscriminatorBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.res = nn.Conv2d(dim, dim_out, 1)

        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding = 1),
            leaky_relu(),
            nn.Conv2d(dim_out, dim_out, 3, padding = 1),
            leaky_relu()
        )

        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        res = self.res(x)
        x = self.net(x)
        x = res + x
        return self.down(x)

class Discriminator(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        resolutions = math.log2(image_size)
        assert resolutions.is_integer(), 'image size must be a power of 2'
        resolutions = int(resolutions)
        layers = resolutions - 1

        init_chan = 64
        max_chan = 400
        chans = list(reversed(list(map(lambda t: 2 ** (11 - t), range(layers)))))
        chans = list(map(lambda n: min(max_chan, n), chans))
        chans = [init_chan, *chans]
        final_chan = chans[-1]

        self.layers = nn.ModuleList([])
        for in_chan, out_chan in zip(chans[:-1], chans[1:]):
            self.layers.append(DiscriminatorBlock(
                dim = in_chan,
                dim_out = out_chan
            ))

        self.initial_conv = nn.Sequential(nn.Conv2d(3, init_chan, 1), leaky_relu())
        self.final_conv = nn.Conv2d(final_chan, 1, 2)

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)
