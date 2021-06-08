"""
This script trains sentence transformers with a triplet loss function.

As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.
"""

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime
from zipfile import ZipFile

from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.datasets import NoDuplicatesDataLoader

import csv
import logging
import os
import sys

sys.path.insert(0, '../../src-py/')
import track_1_kp_matching
from KeyPointEvaluator import KeyPointEvaluator

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'distilbert-base-uncased'

def load_batch_hard_dataset(dataset_path, data_file_suffix):

    
    logger.info("Read Triplet train dataset")
    train_examples = []
    with open(os.path.join(dataset_path, 'training_df_{}.csv'.format(data_file_suffix)), encoding="utf-8") as fIn:
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            #print(row)
            train_examples.append(InputExample(texts=[row['text']], label=int(row['label'])))


    logger.info("Read Triplet dev dataset")
    dev_examples = []
    with open(os.path.join(dataset_path, 'valid_df_{}.csv'.format(data_file_suffix)), encoding="utf-8") as fIn:
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            #print(row)
            dev_examples.append(InputExample(texts=[row['argument'], row['pos_kp'], row['neg_kp']]))

            if len(dev_examples) >= 1000:
                break
              
              
    return train_examples, dev_examples


def train_hard_triplet_modle(dataset_path, output_path, model_name, training_loss='BatchAllTripletLoss', num_epochs=3, train_batch_size=16, data_file_suffix=''):
    
    train_examples, dev_examples = load_batch_hard_dataset(dataset_path, data_file_suffix)
    # We create a special dataset "SentenceLabelDataset" to wrap out train_set
    # It will yield batches that contain at least two samples with the same label
    train_data_sampler = SentenceLabelDataset(train_examples)
    train_dataloader = DataLoader(train_data_sampler, batch_size=train_batch_size, drop_last=True)
    dev_evaluator = TripletEvaluator.from_input_examples(dev_examples, name='dev')

    # Load pretrained model
    logging.info("Load model")
    #model = SentenceTransformer(model_name)

    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    

    ### Triplet losses ####################
    ### There are 4 triplet loss variants:
    ### - BatchHardTripletLoss
    ### - BatchHardSoftMarginTripletLoss
    ### - BatchSemiHardTripletLoss
    ### - BatchAllTripletLoss
    #######################################

    if training_loss =='BatchAllTripletLoss':
        train_loss = losses.BatchAllTripletLoss(model=model)
    
    if training_loss =='BatchHardTripletLoss':
        train_loss = losses.BatchHardTripletLoss(model=model)
    
    if training_loss =='BatchHardSoftMarginTripletLoss':
        train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
    
    if training_loss =='BatchSemiHardTripletLoss':
        train_loss = losses.BatchSemiHardTripletLoss(model=model)

#     if training_loss == 'CustBatchAllTripletLoss':
#         print('Using the CutBatchAllTripletLoss')
#         train_loss = losses.CustBatchAllTripletLoss(model=model)


    logging.info("Performance before fine-tuning:")
    dev_evaluator(model)

    warmup_steps = int(len(train_dataloader) * num_epochs  * 0.1)  # 10% of train data

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=output_path,
    )


def train_model(dataset_path, eval_data_path, subset_name, output_path, model_name, num_epochs=3, train_batch_size=16, model_suffix='', data_file_suffix='', max_seq_length=256, 
                add_special_token=False, loss='Triplet'):
    ### Configure sentence transformers for training and train on the provided dataset
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    output_path = output_path+model_name+ "-" + model_suffix + "-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    word_embedding_model = models.Transformer(model_name)
    word_embedding_model.max_seq_length = max_seq_length
    
    if add_special_token:
        word_embedding_model.tokenizer.add_tokens(['<SEP>'], special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    logger.info("Read Triplet train dataset")
    train_examples = []
    with open(os.path.join(dataset_path, 'training_df_{}.csv'.format(data_file_suffix)), encoding="utf-8") as fIn:
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if loss == 'ContrastiveLoss':
                train_examples.append(InputExample(texts=[row['argument'], row['keypoint']], label=int(row['label'])))
            else:
                train_examples.append(InputExample(texts=[row['anchor'], row['pos'], row['neg']], label=0))



    if loss == 'MultipleNegativesRankingLoss':
        # Special data loader that avoid duplicates within a batch
        train_dataloader = NoDuplicatesDataLoader(train_examples, shuffle=False, batch_size=train_batch_size)
        # Our training loss
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif loss == 'ContrastiveLoss':
        train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=train_batch_size)
        train_loss = losses.ContrastiveLoss(model)
    else:
        train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model)
    

    evaluator = KeyPointEvaluator.from_eval_data_path(eval_data_path, subset_name, add_special_token, name='dev', show_progress_bar=False)


    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data


    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=500,
              warmup_steps=warmup_steps,
              output_path=output_path)