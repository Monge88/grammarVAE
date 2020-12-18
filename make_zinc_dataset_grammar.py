from __future__ import print_function
import argparse
import nltk
import pdb
import zinc_grammar
import numpy as np
import h5py
import molecule_vae
from tqdm import tqdm
import pickle

#f = open('data/250k_rndm_zinc_drugs_clean.smi','r')
#f = open('data/qm9_19_train.smi','r')
with open('data/QM9_STAR.pkl', 'rb') as data:
    f = pickle.load(data)
L = list(f.loc[(f['number_of_atoms']==19, 'SMILES_B3LYP'])


MAX_LEN = 70
NCHARS = len(zinc_grammar.GCFG.productions()) # NCHARS = 76

def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(zinc_grammar.GCFG.productions()):
        prod_map[prod] = ix  # mapeia cada regra de produção a um valor: {smile -> chain: 0}
    tokenize = molecule_vae.get_zinc_tokenizer(zinc_grammar.GCFG)
    tokens = map(tokenize, smiles)  # substitui alguns dos símbolos com len(simbolo) > 1
    parser = nltk.ChartParser(zinc_grammar.GCFG)
    parse_trees = [parser.parse(t).__next__() for t in tokens]  # Alterei .next() para __next__()
    productions_seq = [tree.productions() for tree in parse_trees]  # Cria a árvore com as regras de produção para cada smile string
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]  # Mapeia cada regra de produção com seu índice em GCFG
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    for i in range(len(indices)):  # Alterei xrange para range
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot

    
OH = np.zeros((len(L),MAX_LEN,NCHARS))
for i in range(0, len(L), 100):
    print('Processing: i=[' + str(i) + ':' + str(i+100) + ']')
    onehot = to_one_hot(L[i:i+100])
    OH[i:i+100,:,:] = onehot
    

#h5f = h5py.File('data/zinc_grammar_dataset.h5','w')
#h5f = h5py.File('data/zinc_grammar_dataset_qm9_19_train.h5','w')
h5f = h5py.File('data/qm9_star.h5','w')
h5f.create_dataset('data', data=OH)
h5f.close()
