import math
from pathlib import Path
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from tqdm import trange
from PIL import Image
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as T

from pi_gan_pytorch.coordconv import CoordConv
from einops import rearrange, repeat

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# helper

def exists(val):
    return val is not None

def leaky_relu(p = 0.2):
    return nn.LeakyReLU(p)

def to_value(t):
    return t.clone().detach().item()

def get_module_device(module):
    return next(module.parameters()).device

# losses

def loss_fn(u):
    return -torch.log(1 + torch.exp(-u))

def gradient_penalty(images, output, weight = 10):
    batch_size, device = images.shape[0], images.device
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    l2 = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
    return weight * l2

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

class SirenGenerator(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_hidden,
        siren_num_layers = 6
    ):
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
    def __init__(
        self,
        *,
        image_size,
        dim,
        dim_hidden,
        siren_num_layers
    ):
        super().__init__()
        self.dim = dim

        self.G = SirenGenerator(
            dim = dim,
            dim_hidden = dim_hidden,
            siren_num_layers = siren_num_layers
        )

        self.set_image_size(image_size)

    def set_image_size(self, image_size):
        device = get_module_device(self)
        self.image_size = image_size

        tensors = [torch.linspace(-1, 1, steps = image_size), torch.linspace(-1, 1, steps = image_size)]
        coors = torch.stack(torch.meshgrid(*tensors), dim=-1)
        coors = rearrange(coors, 'h w c -> (h w) c')
        self.register_buffer('coors', coors.to(device))

    def forward(self, x, ray_direction):
        assert ray_direction.shape[-1] == 2, 'ray direction should have a dimension of 2'
        device, b = x.device, x.shape[0]
        coors = repeat(self.coors, 'n c -> b n c', b = b).float()
        ray_direction = repeat(ray_direction, 'b c -> b n c', n = coors.shape[1])
        rgb, alpha = self.G(x, ray_direction, coors) # not sure what to do with alpha
        return rearrange(rgb, 'b (h w) c -> b c h w', h = self.image_size)

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
        assert math.log2(init_resolution).is_integer(), 'initial resolution must be power of 2'

        resolutions = int(resolutions)
        layers = resolutions - 1

        chans = list(reversed(list(map(lambda t: 2 ** (11 - t), range(layers)))))
        chans = list(map(lambda n: min(max_chan, n), chans))
        chans = [init_chan, *chans]
        final_chan = chans[-1]

        self.from_rgb_layers = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        self.image_size = image_size
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

    def increase_resolution_(self):
        if self.resolution >= self.image_size:
            return

        self.alpha += self.alpha + (1 - self.alpha)
        self.iterations.fill_(0.)
        self.resolution *= 2

    def update_iter_(self):
        self.iterations += 1
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
        return out.sigmoid()

# pi-GAN class

class piGAN(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        dim,
        generator_dim_hidden = 256,
        siren_num_layers = 6,
        add_layer_iters = 10000
    ):
        super().__init__()
        self.dim = dim

        self.G = Generator(
            image_size = image_size,
            dim = dim,
            dim_hidden = generator_dim_hidden,
            siren_num_layers = siren_num_layers

        )

        self.D = Discriminator(
            image_size = image_size,
            add_layer_iters = add_layer_iters
        )

# dataset

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        transparent = False,
        aug_prob = 0.,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'
        self.create_transform(image_size)

    def create_transform(self, image_size):
        self.transform = T.Compose([
            T.Lambda(partial(resize_to_minimum_size, image_size)),
            T.CenterCrop(image_size),
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer

def sample_generator(G, batch_size):
    dim = G.dim
    rand_latents = torch.randn(batch_size, dim).cuda()
    rand_ray_dir = torch.randn(batch_size, 2).cuda()
    return G(rand_latents, rand_ray_dir)

class Trainer(nn.Module):
    def __init__(
        self,
        *,
        gan,
        folder,
        add_layers_iters = 10000,
        batch_size = 1,
        gradient_accumulate_every = 4,
        sample_every = 100,
        log_every = 10,
        num_train_steps = 50000,
        lr_gen = 5e-5,
        lr_discr = 4e-4,
        target_lr_gen = 1e-5,
        target_lr_discr = 1e-4,
        lr_decay_span = 10000
    ):
        super().__init__()
        gan.D.add_layer_iters = add_layers_iters
        self.add_layers_iters = add_layers_iters

        self.gan = gan.cuda()

        self.optim_D = Adam(self.gan.D.parameters(), betas=(0, 0.9), lr = lr_discr)
        self.optim_G = Adam(self.gan.G.parameters(), betas=(0, 0.9), lr = lr_gen)

        D_decay_fn = lambda i: max(1 - i / lr_decay_span, 0) + (target_lr_discr / lr_discr) * min(i / lr_decay_span, 1)
        G_decay_fn = lambda i: max(1 - i / lr_decay_span, 0) + (target_lr_gen / lr_gen) * min(i / lr_decay_span, 1)

        self.sched_D = LambdaLR(self.optim_D, D_decay_fn)
        self.sched_G = LambdaLR(self.optim_G, G_decay_fn)

        self.iterations = 0
        self.batch_size = batch_size
        self.num_train_steps = num_train_steps

        self.log_every = log_every
        self.sample_every = sample_every
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = ImageDataset(folder = folder, image_size = gan.D.resolution.item())
        self.dataloader = cycle(DataLoader(self.dataset, batch_size = batch_size, shuffle = True))

        self.last_loss_D = 0
        self.last_loss_G = 0

    def step(self):
        D, G, batch_size, dim, accumulate_every = self.gan.D, self.gan.G, self.batch_size, self.gan.dim, self.gradient_accumulate_every

        # set appropriate image size

        if self.iterations % self.add_layers_iters == 0:
            if self.iterations != 0:
                D.increase_resolution_()

            image_size = D.resolution.item()
            G.set_image_size(image_size)
            self.dataset.create_transform(image_size)

        # train discriminator

        D.train()
        loss_D = 0

        for _ in range(accumulate_every):
            images = next(self.dataloader)
            images = images.cuda().requires_grad_()

            real_out = D(images)
            loss_real = loss_fn(-real_out).mean()
            gp = gradient_penalty(images, real_out)

            fake_out = sample_generator(G, batch_size).detach()
            loss_fake = loss_fn(D(fake_out)).mean()

            loss = gp + loss_real + loss_fake
            self.last_loss_gp = to_value(gp)

            (loss / accumulate_every).backward()
            loss_D += to_value(loss) / accumulate_every

        self.last_loss_D = loss_D

        self.optim_D.step()
        self.optim_D.zero_grad()

        # train generator

        G.train()
        loss_G = 0

        for _ in range(accumulate_every):
            fake_out = sample_generator(G, batch_size)
            loss = loss_fn(-D(fake_out)).mean()
            (loss / accumulate_every).backward()
            loss_G += to_value(loss) / accumulate_every

        self.last_loss_G = loss_G

        self.optim_G.step()
        self.optim_G.zero_grad()

        # update schedulers

        self.sched_D.step()
        self.sched_G.step()

        self.iterations += 1
        D.update_iter_()

    def forward(self):
        for _ in trange(self.num_train_steps):
            self.step()

            if self.iterations % self.log_every == 0:
                print(f'I: {self.gan.D.resolution.item()} | D: {self.last_loss_D:.2f} | G: {self.last_loss_G:.2f} | GP: {self.last_loss_gp:.2f}')

            if self.iterations % self.sample_every == 0:
                i = self.iterations // self.sample_every
                imgs = sample_generator(self.gan.G, 4)
                save_image(imgs, f'./{i}.png', nrow = 2)
