import os
import pandas as pd
import shutil

from bert_utils import *
from bert_gap import BertGAP
from gap_model import GAPModel
import tensorflow as tf

def preprocess_embeddings():

    bert_gap = BertGAP(
            bert_model='models/uncased_L-24_H-1024_A-16',
            emb_size=1024,  # BERT Large embeddings size
            seq_len=64,     # Tokens window size to trocess embeddings 
            n_layers=3,     # Number of output embeddings layers to take 
            start_layer=4,  # Start taking embeddings form this output layer
            do_lower_case=True,
            normalize_text=True)

    bert_gap.process_embeddings(
            input_file = 'gap-coreference/gap-test.tsv',
            output_file = 'input/emb_uncased_test.json')

    bert_gap.process_embeddings(
            input_file = 'gap-coreference/gap-validation.tsv',
            output_file = 'input/emb_uncased_validation.json')

    bert_gap.process_embeddings(
            input_file = 'gap-coreference/gap-development.tsv',
            output_file = 'input/emb_uncased_development.json')

    bert_gap.process_embeddings(
            input_file = 'input/test_stage_1.tsv',
            output_file = 'input/emb_uncased_test_stage_1.json')

    bert_gap = BertGAP(
            bert_model='models/cased_L-24_H-1024_A-16',
            emb_size=1024,
            seq_len=64,
            n_layers=3,
            start_layer=4,
            do_lower_case=False,
            normalize_text=True)

    bert_gap.process_embeddings(
            input_file = 'gap-coreference/gap-test.tsv',
            output_file = 'input/emb_cased_test.json')

    bert_gap.process_embeddings(
            input_file = 'gap-coreference/gap-validation.tsv',
            output_file = 'input/emb_cased_validation.json')

    bert_gap.process_embeddings(
            input_file = 'gap-coreference/gap-development.tsv',
            output_file = 'input/emb_cased_development.json')

    bert_gap.process_embeddings(
            input_file = 'input/test_stage_1.tsv',
            output_file = 'input/emb_cased_test_stage_1.json')


def preprocess():
    
    # Create necessary dirs and downloads BERT and ELMo
    #os.system('sh preparation.sh')
    
    # Create test/train DFs
    train_dfs = []
    for filename in ['gap-coreference/gap-test.tsv', 
                     'gap-coreference/gap-validation.tsv']:
        df = pd.read_csv(filename, sep='\t')
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df.to_csv('input/train.tsv', sep='\t', index=False)
    shutil.copyfile('gap-coreference/gap-development.tsv', 'input/test.tsv')
    

    # Create features
    #os.system('python3 scripts/build_features.py')

    # Create embeddings
    #preprocess_embeddings()


def train_model():
    model = GAPModel()

    # Embeddings .json files computed in preprocess_embeddings()
    embeddings = [
        {
            'train': [
                'input/emb_uncased_test.json',
                'input/emb_uncased_validation.json'
                ],
            'test': ['input/emb_uncased_test_stage_1.json', ]
        },
        {
            'train': [
                'input/emb_cased_test.json',
                'input/emb_cased_validation.json'
                ],
            'test': ['input/emb_cased_test_stage_1.json', ]
        },
    ]

    # Precomputed features .tsv files
    features = {
        'train': ['features/train_features.tsv', ],
        'train_ids': ['input/train.tsv', ],
        'test': ['features/test_features.tsv', ],
        'test_ids': ['input/test.tsv', ],
    }

    # Input dataset files to get labels.
    input_files = {
        'train': ['gap-coreference/gap-test.tsv', 'gap-coreference/gap-validation.tsv'],
        'test': ['gap-coreference/gap-development.tsv', ]
    }

    model.train_model(embeddings=embeddings, features=features, input_files=input_files)


def main():
    preprocess()
    train_model()


if __name__ == '__main__':
    main()
