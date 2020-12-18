import sys
import molecule_vae
import numpy as np
import argparse
import train_zinc as t
import matplotlib.pyplot as plt
import h5py
import pickle
from sklearn.decomposition import PCA

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--epochs', type=int, metavar='N', default=None,    
                        help='Batch size to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=None,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch', type=int, metavar='N', default=None,
                        help='The size of the batch.')
    return parser.parse_args()


args = get_arguments()
# 1. load grammar VAE
grammar_weights = "results/" + "zinc_vae_grammar_L" + str(args.latent_dim) + "_E" + str(args.epochs) + "_B" + str(args.batch) + "_val.hdf5"

grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)

# 2. let's encode and decode some example SMILES strings
#L = []
# with open('data/qm9_19.smi','r') as data:
#     for line in data:
#         line = line.strip()
#         L.append(line)

with open('data/QM9_STAR.pkl', 'rb') as data:
    f = pickle.load(data)
L = list(f.loc[(f['number_of_atoms']==19, 'SMILES_B3LYP'])



# z: encoded latent points
# NOTE: this operation returns the mean of the encoding distribution
# if you would like it to sample from that distribution instead
# replace line 83 in molecule_vae.py with: return self.vae.encoder.predict(one_hot)
z1 = grammar_model.encode(L)

# # mol: decoded SMILES string
# # NOTE: decoding is stochastic so calling this function many
# # times for the same latent point will return different answers
d1 = grammar_model.decode(z1)

for mol,real in zip(d1[:10],L[:10]):
    print(f'\nsample:{mol}\n real:{real}')
    
if args.latent_dim == 2:
    print()
    plt.figure(figsize=(8,8))
    plt.scatter(z1[:, 0], z1[:, 1], alpha=0.5, c='orange', edgecolors='k')
    plt.title('Molecules distribution in a 2D latent space')
    plt.show()
    
else:
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(z1)
    print(reduced_data)
    plt.figure(figsize=(16,12))
    plot_counter = 1
    for prop in df.iloc[:,12:17].columns:
        plt.subplot(int(np.ceil(len(df.iloc[:,12:17].columns)/2)), 3, plot_counter)
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df[prop], cmap='viridis')
        plt.colorbar()
        plt.title(prop)
        plot_counter += 1
    plt.tight_layout()
    plt.show()
    

# # 3. the character VAE (https://github.com/maxhodak/keras-molecules)
# # works the same way, let's load it
# char_weights = "results/" + "zinc_vae_grammar_L" + str(args.latent_dim) + "_E" + str(args.epochs) + "_val.hdf5"
# char_model = molecule_vae.ZincCharacterModel(char_weights)

# # 4. encode and decode
# z2 = char_model.encode(smiles)
# for mol in char_model.decode(z2):
#     print(mol)
