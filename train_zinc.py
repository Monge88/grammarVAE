from __future__ import print_function

import gc
import argparse
import os
import h5py
import numpy as np

from models.model_zinc import MoleculeVAE
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import h5py
import zinc_grammar as G
import pdb

# importa as regras de criaÃ§Ã£o
rules = G.gram.split('\n')


MAX_LEN = 70
DIM = len(rules)
LATENT = 56
EPOCHS = 100
BATCH = 64


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--load_model', type=str, metavar='N', default="")
    parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--batch', type=int, metavar='N', default=BATCH,    # Adicionei para controlar o tamanho do batch
                        help='Batch size to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
                        help='Dimensionality of the latent representation.')

                        
    return parser.parse_args()


def main():
    # 0. load dataset
    #h5f = h5py.File('data/zinc_grammar_dataset_qm9.h5', 'r')
    #h5f = h5py.File('data/zinc_grammar_dataset_qm9_19.h5', 'r')
    h5f = h5py.File('data/qm9_star.h5', 'r')
    data = h5f['data'][:]
    h5f.close()
    
    # 1. split into train/test, we use test set to check reconstruction error and the % of
    # samples from prior p(z) that are valid
        
    # No original, XTE = data[0:5000] e XTR = data[5000:], ou seja, cerca 10% de dados para teste
    # Como data pode mudar de quantidade (para que seja possivel rodar no meu computador), alterei 
    # para que as amostras fossem obtidas de acordo com uma certa porcentagem

    np.random.seed(1)
    # 2. get any arguments and define save file, then create the VAE model
    args = get_arguments()
    print('L='  + str(args.latent_dim) + ' E=' + str(args.epochs))
    model_save = 'results/zinc_vae_grammar_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_B' + str(args.batch) + '_val.hdf5'
    print(model_save)
    model = MoleculeVAE()
    print(args.load_model)
    
    # 3. if this results file exists already load it
    if os.path.isfile(args.load_model):
        print('loading!')
        model.load(rules, args.load_model, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        print('making new model')
        model.create(rules, max_length=MAX_LEN, latent_rep_size = args.latent_dim)
        

    # 4. only save best model found on a 10% validation set
    checkpointer = ModelCheckpoint(filepath = model_save,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)
    
  
    # 5. fit the vae
    model.autoencoder.fit(
        data,
        data,
        shuffle = True,
        epochs = args.epochs,  # Alterei nb_epochs para epochs
        batch_size = args.batch,
        callbacks = [checkpointer,reduce_lr],
        validation_split = 0.05)


if __name__ == '__main__':
    main()
