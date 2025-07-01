

import torch 
from torch import nn
import numpy as np
import torch.nn.functional as F

"""

Credits to Barney Emmens for the Angular Spectrum Method (ASM) function code. 

The code for the spectral layer was inspired from https://github.com/PORPHURA/GedankenNet/blob/main/GedankenNet_Phase/networks/fno.py. 
The original implementation of a Fourier Neural Operator (incl. spectral layers) can be found at 
https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/spectral_convolution.py .

"""


def ASM(P0,dx,z, k):
    # Zero_padding to hint to fft that field is consistent to infinities


    Nk = 2**int(np.ceil(np.log2(P0.shape[2]))+1)
    kmax = 2*np.pi/dx

    kv = torch.fft.fftfreq(Nk)*kmax # Compute the spatial frequencies
    kx, ky = torch.meshgrid(kv, kv, indexing='ij')
    
    kz = torch.sqrt((k**2 - kx**2 - ky**2).to(torch.complex64))# Allow for complex values
    
    H = torch.exp(-1j*kz*z)

    D = (Nk-1)*dx
    kc = k*torch.sqrt(torch.tensor(0.5*(D**2)/(0.5*D**2 + z**2))) # What is kc and why is it needed instead of just k?
    H[torch.sqrt(kx**2 + ky**2) > kc] = 0 # Wavelengths greater than kc cannot propogate

    #P0_padded = 
    P0_fourier = torch.fft.fft2(P0,[Nk,Nk]) # Compute the 2D Fourier Transform of the input field
    P_z_fourier = P0_fourier * H

    P_z = torch.fft.ifft2(P_z_fourier,[Nk,Nk]) # Compute the inverse 2D Fourier Transform of the field
    P_z = P_z[..., :P0.shape[2], :P0.shape[3]]
    # P_z *= np.exp(-1j*pi/2) # Phase fudge factor to match Huygens

    return P_z


nconv = 64

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.modes = modes
        scale = 1 / in_channels
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes // 2 + 1, 2))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(batchsize, x.size(1), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        x_ft = x_ft[:, :, :self.modes, :self.modes]
        w = torch.view_as_complex(self.weight)
        out_ft[:, :, :self.modes, :self.modes] = self.compl_mul2d(x_ft, w)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))
        return x

class SpectralBlock(nn.Module):
    def __init__(self, channels, modes=16):
        super().__init__()
        self.spec_conv = SpectralConv2d(channels, channels, modes)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.prelu = nn.PReLU(channels) #try GELU?

    def forward(self, x):
        x_spec = self.spec_conv(x)
        x_conv = self.conv(x)
        return self.prelu(x + x_spec + x_conv)

class In(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x) 

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Up(nn.Module): 
   
    def __init__(self, in_channels, out_channels): 
        super().__init__()


        self.deconv= nn.Sequential(
    
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
    
    def forward(self, x):
        return self.deconv(x)
    
class Out(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)            
        

    def forward(self, x):
        return self.out(x)

class recon_model(nn.Module):

  def __init__(self):
    super(recon_model, self).__init__()

    self.inc = In(1, nconv) 
    self.down1 = Down(nconv, nconv*2)
    self.down2= Down(nconv*2, nconv*4)
  
    self.spec_block = SpectralBlock(nconv * 4, modes=16)

    self.up2 = Up(nconv*4,  nconv*2)
    self.up3 = Up(nconv*2,  nconv)

    self.outc = Out(nconv, 1)            


  def forward(self,x):

    x = self.inc(x)

    x = self.down1(x)
    x = self.down2(x)

    x = self.spec_block(x)

    x = self.up2(x)
    x = self.up3(x)

    logits = self.outc(x)

    ############# FORWARD MODEL #############

    P0_phase = logits
    P0_phase = torch.tanh(P0_phase) # tanh activation (-1 to 1) 
    P0_phase = P0_phase*np.pi # restore to (-pi, pi) range

    # no_elements_per_side = 11
    # phase_elem = F.interpolate(P0_phase, size=(no_elements_per_side, no_elements_per_side), mode='area')
    # P0_phase = F.interpolate(phase_elem, size=(64, 64), mode='nearest')

    amp = 1
   
    #Create the complex number
    P0 = torch.complex(amp*torch.cos(P0_phase),amp*torch.sin(P0_phase))


    c_w = 1480
    c_p = 2340

    #### wave parameters ####
    f = 1e6
    wavelength = c_w/f
    k = 2*np.pi*f/c_w

    #### Source Parameters ####
    element_width = 3e-3
    kerf = 0.1e-3
    N_elements_per_side = 7
    pitch = element_width + kerf
    aperture = N_elements_per_side*pitch - 2*kerf

    dx = wavelength/4
    z = 1.5*aperture


    P_z_ASM = ASM(P0, dx, z, k)
    P_z_magn = torch.abs(P_z_ASM)

    return P_z_magn, P0_phase
  


