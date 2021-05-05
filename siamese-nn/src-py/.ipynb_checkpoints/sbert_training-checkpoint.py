"""
This script trains sentence transformers with a triplet loss function.

As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.
"""

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime
from zipfile import ZipFile

import csv
import logging
import os

dataset_path = '/workspace/ceph_data/keypoint-analysis-sharedtask/siamese-data/'
output_path = '/workspace/ceph_data/keypoint-analysis-sharedtask/siamese-models/'

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'distilbert-base-uncased'

        
### Create a torch.DataLoader that passes training batch instances to our model
train_batch_size = 16
output_path = "output/training-keypoint-analysis-"+model_name+"-"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
num_epochs = 3


### Configure sentence transformers for training and train on the provided dataset
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


logger.info("Read Triplet train dataset")
train_examples = []
with open(os.path.join(dataset_path, 'training_df.csv'), encoding="utf-8") as fIn:
    reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        train_examples.append(InputExample(texts=[row['argument'], row['pos_kp'], row['neg_kp']], label=0))



train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.TripletLoss(model=model)

logger.info("Read Triplet dev dataset")
dev_examples = []
with open(os.path.join(dataset_path, 'valid_df.csv'), encoding="utf-8") as fIn:
    reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        dev_examples.append(InputExample(texts=[row['argument'], row['pos_kp'], row['neg_kp']]))

        if len(dev_examples) >= 1000:
            break

evaluator = TripletEvaluator.from_input_examples(dev_examples, name='dev')


warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=output_path)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

logger.info("Read test examples")
test_examples = []
with open(os.path.join(dataset_path, 'valid_df.csv'), encoding="utf-8") as fIn:
    reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        test_examples.append(InputExample(texts=[row['argument'], row['pos_kp'], row['neg_kp']]))


model = SentenceTransformer(output_path)
test_evaluator = TripletEvaluator.from_input_examples(test_examples, name='test')
test_evaluator(model, output_path=output_path)
