from pathlib import Path
from data.data_pipe import load_bin
from torchvision import transforms as trans

# Validation data could be downloaded from:
# https://github.com/deepinsight/insightface/wiki/Dataset-Zoo -->> They come with any of the training datasets.
# https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0 --> eg MS1M-ArcFace
# Make sure it is in the rec_path = Path('data') / Path('faces_webface') path!

if __name__ == '__main__':
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    rec_path = Path('data') / Path('faces_webface')

    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw']

    for i in range(len(bin_files)):
        load_bin(rec_path / (bin_files[i] + '.bin'), rec_path / bin_files[i], test_transform)