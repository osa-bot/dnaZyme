import random
import numpy as np
import pandas as pd

from rdkit.Chem import rdMolDescriptors
from rdkit import Chem

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Model


aa_dict: dict[str, str] = {
    "dA": r"O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)C[C@@H]3O",  # DNA
    "dT": r"CC1=CN(C(=O)NC1=O)C2CC(C(O2)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O",
    "dG": r"O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n1cnc2c1NC(=N/C2=O)\N)C[C@@H]3O",
    "dC": r"C1[C@@H]([C@H](O[C@H]1N2C=CC(=NC2=O)N)CO[P@@](=O)(O)O[P@@](=O)(O)OP(=O)(O)O)O",
    "rA": r"c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N",  # RNA
    "rU": r"C1=CN(C(=O)NC1=O)C2C(C(C(O2)COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O",
    "rG": r"C1=NC2=C(N1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N",
    "rC": r"c1cn(c(=O)nc1N)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O",
    "A": "CC(C(=O)O)N",  # protein
    "R": "C(CC(C(=O)O)N)CN=C(N)N",
    "N": "C(C(C(=O)O)N)C(=O)N",
    "D": "C(C(C(=O)O)N)C(=O)O",
    "C": "C(C(C(=O)O)N)S",
    "Q": "C(CC(=O)N)C(C(=O)O)N",
    "E": "C(CC(=O)O)C(C(=O)O)N",
    "G": "C(C(=O)O)N",
    "H": "C1=C(NC=N1)CC(C(=O)O)N",
    "I": "CCC(C)C(C(=O)O)N",
    "L": "CC(C)CC(C(=O)O)N",
    "K": "C(CCN)CC(C(=O)O)N",
    "M": "CSCCC(C(=O)O)N",
    "F": "C1=CC=C(C=C1)CC(C(=O)O)N",
    "P": "C1CC(NC1)C(=O)O",
    "S": "C(C(C(=O)O)N)O",
    "T": "CC(C(C(=O)O)N)O",
    "W": "C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N",
    "Y": "C1=CC(=CC=C1CC(C(=O)O)N)O",
    "V": "CC(C)C(C(=O)O)N",
    "O": "CC1CC=NC1C(=O)NCCCCC(C(=O)O)N",
    "U": "C(C(C(=O)O)N)[Se]",
}


def generate_rdkit_descriptors(normalize: tuple = (-1, 1), monomer_dict_=aa_dict):
    """
    Generates and normalizes RDKit descriptors for a set of molecules.

        This method calculates a variety of molecular descriptors using RDKit,
        normalizes them to a specified range, and returns the results in a Pandas DataFrame.

        Args:
            normalize: The desired range for normalization (min, max). Defaults to (-1, 1).
            monomer_dict_: A dictionary where keys are monomer identifiers and values are SMILES strings.

        Returns:
            pd.DataFrame: A DataFrame containing the normalized RDKit descriptors,
                          with columns named after the descriptor names and rows indexed by
                          the monomer identifiers from the input dictionary.
    """
    descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    num_descriptors = len(descriptor_names)

    descriptors_set = np.empty((0, num_descriptors), float)

    for _, value in monomer_dict_.items():
        molecule = Chem.MolFromSmiles(value)
        descriptors = np.array(get_descriptors.ComputeProperties(molecule)).reshape(
            (-1, num_descriptors)
        )
        descriptors_set = np.append(descriptors_set, descriptors, axis=0)

    sc = MinMaxScaler(feature_range=normalize)
    scaled_array = sc.fit_transform(descriptors_set)
    return pd.DataFrame(
        scaled_array, columns=descriptor_names, index=list(monomer_dict_.keys())
    )


def filter_sequences(
    sequences: [list, pd.DataFrame],
    max_length: int = 96,
    sequences_column_name: str = None,
    shuffle_seqs: bool = True,
    aa_dict_=aa_dict,
):
    """
    Filters a list or Pandas DataFrame of sequences based on length and allowed amino acids.

        Args:
            sequences: A list of strings (amino acid sequences) or a Pandas DataFrame
                containing sequences in a specified column.
            max_length: The maximum acceptable length for a sequence. Sequences longer than this are discarded.
            sequences_column_name: The name of the column in the DataFrame that contains the sequences.
                Required if `sequences` is a DataFrame.
            shuffle_seqs: Whether to randomly shuffle the filtered sequences.
            aa_dict_: A dictionary mapping amino acids to their representations. Used to filter out
                sequences containing invalid amino acids.

        Returns:
            list or pd.DataFrame: A list of filtered sequences (if input was a list) or a DataFrame
            containing only the filtered sequences (if input was a DataFrame).
    """
    if type(sequences) is list:
        all_seqs = [seq.upper() for seq in sequences if len(seq) <= max_length]
        all_seqs = list(dict.fromkeys(all_seqs))
        filtered_seqs = [x for x in all_seqs if set(x).issubset(set(aa_dict_.keys()))]
        if shuffle_seqs:
            filtered_seqs = random.sample(filtered_seqs, len(filtered_seqs))
        return filtered_seqs

    elif type(sequences) is pd.DataFrame:
        sequences[sequences_column_name] = sequences[sequences_column_name].map(
            lambda x: x.replace(" ", "")
        )
        sequences[sequences_column_name] = sequences[sequences_column_name].str.upper()
        sequences = sequences[
            sequences[sequences_column_name].apply(lambda x: len(x) <= max_length)
        ]
        peptide_subs = sequences[
            sequences[sequences_column_name].apply(
                lambda x: set(x).issubset(set(aa_dict_.keys()))
            )
        ]
        if shuffle_seqs:
            peptide_subs = peptide_subs.sample(frac=1)
        print(peptide_subs)
        return peptide_subs


def seq_to_matrix_(sequence, polymer_type, descriptors, num):
    """
    Converts a sequence into a matrix representation using provided descriptors.

        This function iterates through the input sequence and retrieves corresponding
        descriptors from a lookup table. It then constructs a matrix where each row
        represents the descriptor vector for an amino acid/nucleotide in the sequence.
        Finally, it pads the matrix with -1 values if its width is less than 'num'.

        Args:
            sequence: The input sequence (e.g., protein or DNA sequence).
            polymer_type:  The type of polymer ('DNA' supported currently).
            descriptors: A data structure containing descriptors for each amino acid/nucleotide.
            num: The desired number of columns in the output matrix.

        Returns:
            tf.Tensor: A TensorFlow tensor representing the sequence as a matrix, padded to 'num' columns if necessary.  Returns None if an invalid polymer type is provided.
    """
    if polymer_type == "DNA":
        prefix = "d"
    else:
        print("Wrong polymer type")
        return

    rows = descriptors.shape[1]
    seq_matrix = tf.zeros(shape=[0, rows])
    for aa in sequence:
        aa_params = tf.constant(descriptors.loc[prefix + aa], dtype=tf.float32)
        descriptors_array = tf.expand_dims(aa_params, axis=0)
        seq_matrix = tf.concat([seq_matrix, descriptors_array], axis=0)
    seq_matrix = tf.transpose(seq_matrix)
    shape = seq_matrix.get_shape().as_list()[1]
    if shape < num:
        paddings = tf.constant([[0, 0], [0, num - shape]])
        add_matrix = tf.pad(
            seq_matrix, paddings=paddings, mode="CONSTANT", constant_values=-1
        )

        return add_matrix

    return seq_matrix


def encoding(sequences_list, polymer_type, descriptors, num):
    """
    Encodes a list of sequences into a TensorFlow tensor.

        This method iterates through a list of sequences, converts each sequence
        into a matrix representation using the seq_to_matrix_ function, and then
        concatenates these matrices along the first axis to create a single
        TensorFlow tensor.  Prints progress updates every 3200 sequences.

        Args:
            sequences_list: The list of sequences to encode.
            polymer_type: Specifies the type of polymer for sequence encoding.
            descriptors: Descriptors used in the sequence-to-matrix conversion.
            num: An integer parameter used in the sequence-to-matrix conversion.

        Returns:
            tf.Tensor: A TensorFlow tensor containing the encoded sequences.
    """
    container = []
    for i, sequence in enumerate(sequences_list):
        if i % 3200 == 0:
            print(i * 100 / len(sequences_list), " %")

        seq_matrix = tf.expand_dims(
            seq_to_matrix_(
                sequence=sequence,
                polymer_type=polymer_type,
                descriptors=descriptors,
                num=num,
            ),
            axis=0,
        )
        container.append(seq_matrix)
    encoded_seqs = tf.concat(container, axis=0)

    return encoded_seqs


def generate_latent_representations(encoded_sequences, path_to_model_folder=""):
    """
    Generates latent representations from encoded sequences using a pre-trained model.

        Args:
            encoded_sequences: The input sequences to encode.
            path_to_model_folder: Path to the folder containing the trained Keras model.
                If empty, it's assumed the model is in the current directory.

        Returns:
            numpy.ndarray: A numpy array representing the latent representations of the input sequences.
    """
    trained_model = tf.keras.models.load_model(path_to_model_folder)

    layer_name = "Latent"
    intermediate_layer_model = Model(
        inputs=trained_model.input, outputs=trained_model.get_layer(layer_name).output
    )
    latent_representation = intermediate_layer_model.predict(encoded_sequences)
    return latent_representation
