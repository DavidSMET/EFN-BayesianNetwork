import pandas as pd
import urllib.request
import gzip

from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model

from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian
from dag_gflownet.utils.sampling import sample_from_linear_gaussian


def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(filename, 'wb') as f:
        f.write(file_content)
    
    return filename


def get_data(name, args, rng=default_rng()):
    small = ['asia', 'cancer', 'earthquake', 'sachs', 'survey']
    medium = ['alarm', 'barley', 'child', 'insurance', 'mildew', 'water']
    large = ['hailfinder', 'hepar2', 'win95pts']

    if name == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng
        )
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples,
            rng=rng
        )
        score = 'bge'

    elif name == 'sachs_continuous':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
            Path('data/sachs.data.txt')
        )
        data = pd.read_csv(filename, delimiter='\t', dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'

    elif name =='sachs_interventional':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz',
            Path('data/sachs.interventional.txt')
        )
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bde'
    
    elif name in small:
        graph = get_example_model(name)
        filename = get_data_txt(name, 1000)
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bde'
    
    elif name in medium:
        graph = get_example_model(name)
        filename = get_data_txt(name, 5000)
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bic'
            
    elif name in large:
        graph = get_example_model(name)
        filename = get_data_txt(name, 8000)
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bic'
    
    else:
        raise ValueError(f'Unknown graph type: {name}')

    return graph, data, score

def get_data_txt(name, sample_nums = 1000):
    import os
    from pgmpy.sampling import BayesianModelSampling

    str1 = "data/"
    str2 = ".txt"
    file_path = str1 + name + str2
    if os.path.exists(file_path):
        return Path(file_path)

    model = get_example_model(name)
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=sample_nums, seed=42)
    header_line = ' '.join([f'"{col}"' for col in model.nodes()])
    data_lines = samples.to_csv(
        sep=" ",          
        index=False,      
        header=False      
    ).splitlines()

    os.makedirs("data", exist_ok=True)
    with open(file_path, "w") as f:
        f.write(header_line + "\n")
        for line in data_lines:
            f.write(line + "\n")
    return Path(file_path)