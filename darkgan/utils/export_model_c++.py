import click
import torch
import os
from utils.utils import mkdir_in_path, load_model_checkp
import ipdb
from scipy.signal.windows import hann


class GWrappperISTFT(torch.nn.Module):
    def __init__(self, model):
        super(GWrappperISTFT, self).__init__()
        self.model = model
        self.win = torch.from_numpy(hann(2048)).float()

    def forward(self, x):
        with torch.no_grad():
            y = self.model(x)
            y = y.permute(0, 2, 3, 1)
<<<<<<< HEAD
            y = torch.cat([torch.zeros(1, 1, 32, 2), y], dim=1)
            y = torch.istft(y, 2048, hop_length=512, length=16384-1, window=self.win)
=======
            y = torch.cat([torch.zeros(1, 1, 64, 2), y], dim=1)
            y = torch.istft(y, 2048, hop_length=512, length=32768-1, window=self.win)
>>>>>>> master
        return y

class DWrappperSTFT(torch.nn.Module):
    def __init__(self, model):
        super(DWrappperSTFT, self).__init__()
        self.model = model
        self.win = torch.from_numpy(hann(2048)).float()

    def forward(self, x):
        with torch.no_grad():
            x = torch.stft(x, 2048, hop_length=512, window=self.win)
            x = x.permute(0, 3, 1, 2)
            x = x[:, :, 1:, :]
            y = self.model(x)
        
        return y

@click.command()
@click.option('-d', '--dir', type=click.Path(exists=True), required=True, default='')
@click.option('-i', '--iteration', type=int, required=False, default=None)
@click.option('-s', '--scale', type=int, required=False, default=None)
def main(dir, iteration, scale):
    out_path = mkdir_in_path(dir, 'traced_model')
    model, config, model_name = \
        load_model_checkp(dir, iteration, scale)

    # trace G
    G = model.getOriginalG().eval().cpu()
    G = GWrappperISTFT(G)
    slice = model.buildNoiseData(1, skipAtts=True)[0].cpu()
    outG = G(slice)
    out_fileG = f'{model_name}_G.pt'
    out_pathG = os.path.join(out_path, out_fileG)
    
    traced_script_moduleG = torch.jit.trace(G, slice)
    retrace = traced_script_moduleG(slice)
    assert torch.allclose(outG, retrace), 'Found output inconsistency!'

    traced_script_moduleG.save(out_pathG)
    print('Traced saved: {}'.format(out_pathG))

    traced_script_moduleG = torch.jit.load(out_pathG, 'cpu')
    retrace = traced_script_moduleG(slice)
    assert torch.allclose(outG, retrace), 'Found output inconsistency!'

    # trace D
    D = model.getOriginalD().eval().cpu()
    D = DWrappperSTFT(D)
    outD = D(outG)
    out_fileD = f'{model_name}_D.pt'
    out_pathD = os.path.join(out_path, out_fileD)

    traced_script_moduleD = torch.jit.trace(D, outG)
    retrace = traced_script_moduleD(outG)
    assert torch.allclose(outD, retrace), 'Found output inconsistency!'

    traced_script_moduleD.save(out_pathD)
    print('Traced saved: {}'.format(out_pathD))

    traced_script_moduleD = torch.jit.load(out_pathD, 'cpu')
    retrace = traced_script_moduleD(outG)
    assert torch.allclose(outD, retrace), 'Found output inconsistency!'
if __name__ == '__main__':
    main()








