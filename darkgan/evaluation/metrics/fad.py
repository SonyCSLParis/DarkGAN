import subprocess

from utils.utils import GPU_is_available
from utils.utils import list_files_abs_path, mkdir_in_path

from datetime import datetime
import ipdb
import numpy as np
from tqdm import trange



def compute_fad(path_true, path_fake, output_path):
    true_files = list_files_abs_path(path_true, 'wav')
    fake_files = list_files_abs_path(path_fake, 'wav')

    n_files = max(len(true_files), len(fake_files))
    fad_bsize = 5000
    fad = []
    for i in trange(int(np.ceil(n_files/fad_bsize))):
        t_files = np.random.choice(true_files, fad_bsize)
        f_files = fake_files[i*fad_bsize:(i+1)*fad_bsize]
        real_paths_csv = f"{output_path}/real_audio.cvs"
        with open(real_paths_csv, "w") as f:
            for file_path in t_files:
                f.write(file_path + '\n')
        fake_paths_csv = f"{output_path}/fake_audio.cvs"
        with open(fake_paths_csv, "w") as f:
            for file_path in f_files:
                f.write(file_path + '\n')

        fad.append(float(subprocess.check_output(["sh",
                            "shell_scripts/fad.sh",
                            "--real="+real_paths_csv,
                            "--fake="+fake_paths_csv,
                            "--output="+output_path]).decode()[-10:-1]))
        print(fad)
    return np.mean(fad), np.std(fad)


def test(parser, visualisation=None):
    args = parser.parse_args()
    if GPU_is_available:
        device = 'cuda'
    else:
        device = 'cpu'

    output_path = mkdir_in_path(args.dir, "evaluation_metrics")
    output_path = mkdir_in_path(output_path, "fad")
    fad = compute_fad(args.true_path, args.fake_path, output_path)

    with open(f"{output_path}/fad_{datetime.now().strftime('%y_%m_%d')}.txt", "w") as f:
        f.write(str(fad))
        f.close()

    print("FAD={0:.4f}".format(fad))
