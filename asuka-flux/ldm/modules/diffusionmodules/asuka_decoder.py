# pytorch_diffusion + derived decoder
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from ldm.modules.attention import LinearAttention

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, activate_before='none', activate_after='none', upsample_type='deconv'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate_before = activate_before
        self.activate_after = activate_after
        self.upsample_type = upsample_type
        if self.upsample_type == 'deconv':
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            assert self.upsample_type in ['bilinear', 'nearest'], 'upsample {} not implemented!'.format(self.upsample_type)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        if self.activate_before == 'relu':
            x = F.relu(x)
        elif self.activate_before == 'none':
            pass
        else:
            raise NotImplementedError

        if self.upsample_type == 'deconv':
            x = self.deconv(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode=self.upsample_type)
            x = self.conv(x)

        if self.activate_after == 'relu':
            x = F.relu(x)
        elif self.activate_after == 'none':
            pass
        else:
            raise NotImplementedError
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x.to(torch.float32), scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        if XFORMERS_IS_AVAILBLE:
            b, c, h, w = q.shape
            q = q.reshape(b, c, h * w)
            q = q.permute(0, 2, 1)  # b,hw,c
            k = k.reshape(b, c, h * w)
            k = k.permute(0, 2, 1)  # b,hw,c
            v = v.reshape(b, c, h * w)
            v = v.permute(0, 2, 1)  # b,hw,c

            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], 1, c)
                .permute(0, 2, 1, 3)
                .reshape(b * 1, t.shape[1], c)
                .contiguous(),
                (q, k, v),
            )

            # actually compute the attention, what we cannot get enough of
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

            h_ = (
                out.unsqueeze(0)
                .reshape(b, 1, out.shape[1], c)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], c)
            )
            h_ = h_.reshape(b, h, w, c).permute(0, 3, 1, 2)
        else:

            # compute attention
            b, c, h, w = q.shape
            q = q.reshape(b, c, h * w)
            q = q.permute(0, 2, 1)  # b,hw,c
            k = k.reshape(b, c, h * w)  # b,c,hw
            w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w_ = w_ * (int(c) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)

            # attend to values
            v = v.reshape(b, c, h * w)
            w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
            h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Conditionencoder(nn.Module):
    def __init__(self, in_ch, up_layers, downsample_layer='downsample', concat_mask=False, gate_conv=False):
        super().__init__()

        out_channels = []
        for layer in up_layers:
            out_channels.append(layer.out_channels)

        layers = []
        in_ch_ = in_ch
        self.concat_mask = concat_mask
        self.gate_conv = gate_conv
        for l in range(len(out_channels), -1, -1):
            out_ch_ = out_channels[l - 1]
            if gate_conv:
                out_ch_ *= 2
            if l == len(out_channels) or l == len(out_channels) - 1:
                if l == len(out_channels) and concat_mask:
                    layers.append(nn.Conv2d(in_ch_ + 1, out_ch_, kernel_size=3, stride=1, padding=1))
                else:
                    layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=3, stride=1, padding=1))
            else:
                if l == 0:
                    out_ch_ = up_layers[0].in_channels
                    if gate_conv:
                        out_ch_ *= 2
                if isinstance(up_layers[l], UpSample):
                    if downsample_layer == 'downsample':
                        layers.append(DownSample(in_ch_, out_ch_, activate_before='relu', activate_after='none',
                                                 downsample_type='conv', partial_conv=partial_conv))
                    elif downsample_layer == 'conv':
                        layers.append(nn.Conv2d(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            if gate_conv:
                in_ch_ = out_ch_ // 2
            else:
                in_ch_ = out_ch_

        self.layers = nn.Sequential(*layers)
        self.downsample_layer = downsample_layer

    def forward(self, x, mask=None):
        out = {}
        for l in range(len(self.layers)):  # layer in self.layers:
            layer = self.layers[l]
            if l == 0 and self.concat_mask:
                x = layer(torch.cat([x, mask], dim=1))
            else:
                x = layer(x)

            if self.gate_conv:
                [x, gate] = torch.split(x, x.shape[1] // 2, dim=1)
                gate = torch.clamp(1 - torch.tanh(gate), 0, 1)
                x = x * gate

            out[str(tuple(x.shape))] = x  # before activation, because other modules perform activativation first
            if self.downsample_layer == 'conv':
                x = F.relu(x)
        return out


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", concat_mask=False, gate_conv=False, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        up_layers = []
        out_ch_ = 128
        res_ch = 512
        stride = 16
        upsample_type = 'deconv'
        while stride > 1:
            stride = stride // 2
            in_ch_ = out_ch_ * 2
            if out_ch_ > res_ch:
                out_ch_ = res_ch
            if stride == 1:
                in_ch_ = res_ch
            layers_ = []
            layers_.append(UpSample(in_ch_, out_ch_, activate_before='relu', activate_after='none', upsample_type=upsample_type))
            up_layers = layers_ + up_layers
            out_ch_ *= 2
        self.up_layers = nn.Sequential(*up_layers)
        self.encoder = Conditionencoder(
            in_ch=out_ch,
            up_layers=self.up_layers,
            downsample_layer="conv",
            concat_mask=concat_mask,
            gate_conv=gate_conv
        )
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, image, mask):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        masked_image = (1 - mask) * image
        im_x = self.encoder.forward(masked_image, mask)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            h_ = im_x[str(tuple(h.shape))]
            mask_ = F.interpolate(mask, size=h.shape[-2:], mode='area')
            mask_[mask_ > 0] = 1
            h = h * mask_ + h_ * (1 - mask_)
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = h * mask + im_x[str(tuple(h.shape))] * (1 - mask)
        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
