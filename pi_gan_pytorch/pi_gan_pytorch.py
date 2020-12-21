import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from pi_gan_pytorch.coordconv import CoordConv
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

# discriminator

class DiscriminatorBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.res = CoordConv(dim, dim_out, kernel_size = 1)

        self.net = nn.Sequential(
            CoordConv(dim, dim_out, kernel_size = 3, padding = 1),
            leaky_relu(),
            CoordConv(dim_out, dim_out, kernel_size = 3, padding = 1),
            leaky_relu()
        )

        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        res = self.res(x)
        x = self.net(x)
        x = res + x
        return self.down(x)

class Discriminator(nn.Module):
    def __init__(
        self,
        image_size,
        init_chan = 64,
        max_chan = 400,
        init_resolution = 16,
        add_layer_iters = 10000
    ):
        super().__init__()
        resolutions = math.log2(image_size)
        assert resolutions.is_integer(), 'image size must be a power of 2'
        resolutions = int(resolutions)
        layers = resolutions - 1

        chans = list(reversed(list(map(lambda t: 2 ** (11 - t), range(layers)))))
        chans = list(map(lambda n: min(max_chan, n), chans))
        chans = [init_chan, *chans]
        final_chan = chans[-1]

        self.from_rgb_layers = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        self.resolutions = list(map(lambda t: 2 ** (7 - t), range(layers)))

        for resolution, in_chan, out_chan in zip(self.resolutions, chans[:-1], chans[1:]):

            from_rgb_layer = nn.Sequential(
                CoordConv(3, in_chan, kernel_size = 1),
                leaky_relu()
            ) if resolution >= init_resolution else None

            self.from_rgb_layers.append(from_rgb_layer)

            self.layers.append(DiscriminatorBlock(
                dim = in_chan,
                dim_out = out_chan
            ))

        self.final_conv = CoordConv(final_chan, 1, kernel_size = 2)

        self.add_layer_iters = add_layer_iters
        self.register_buffer('alpha', torch.tensor(0.))
        self.register_buffer('resolution', torch.tensor(init_resolution))
        self.register_buffer('iterations', torch.tensor(0.))

    def incr_iter_(self):
        self.iterations += 1

        if self.iterations >= self.add_layer_iters:
            self.alpha.fill_(1.)
            self.iterations.fill_(0.)
            self.resolution *= 2
            return

        if self.alpha == 0:
            return

        self.alpha -= (1 / self.add_layer_iters)
        self.alpha.clamp_(min = 0.)

    def forward(self, img):
        x = img

        for resolution, from_rgb, layer in zip(self.resolutions, self.from_rgb_layers, self.layers):
            if self.resolution < resolution:
                continue

            if self.resolution == resolution:
                x = from_rgb(x)

            if bool(resolution == (self.resolution // 2)) and bool(self.alpha > 0):
                x_down = F.interpolate(img, scale_factor = 0.5)
                x = x * (1 - self.alpha) + from_rgb(x_down) * self.alpha

            x = layer(x)

        out = self.final_conv(x)

        if self.training:
            self.incr_iter_()
        return out

# pi-GAN class

class piGAN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# trainer

class Trainer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x