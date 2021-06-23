from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
from sentence_transformers import util
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from typing import List


logger = logging.getLogger(__name__)

##### METHODS FROM THE KPA EVALUATION FILE ##########

import sys
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import os
import json


def get_ap(df, label_column, top_percentile=0.5):
    top = int(len(df)*top_percentile)
    df = df.sort_values('score', ascending=False).head(top)
    # after selecting top percentile candidates, we set the score for the dummy kp to 1, to prevent it from increasing the precision.
    df.loc[df['key_point_id'] == "dummy_id", 'score'] = 0.99
    return average_precision_score(y_true=df[label_column], y_score=df["score"])

def calc_mean_average_precision(df, label_column):
    precisions = [get_ap(group, label_column) for _, group in df.groupby(["topic", "stance"])]
    return np.mean(precisions)

def evaluate_predictions(merged_df):
    mAP_strict = calc_mean_average_precision(merged_df, "label_strict")
    mAP_relaxed = calc_mean_average_precision(merged_df, "label_relaxed")
    return mAP_strict, mAP_relaxed

def load_kpm_data(gold_data_dir, subset):
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")

    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)

    return arguments_df, key_points_df, labels_file_df


def get_predictions(preds, labels_df, arg_df):
    arg_df = arg_df[["arg_id", "topic", "stance"]]
    predictions_df = load_predictions(preds)
    #make sure each arg_id has a prediction
    predictions_df = pd.merge(arg_df, predictions_df, how="left", on="arg_id")

    #handle arguements with no matching key point
    predictions_df["key_point_id"] = predictions_df["key_point_id"].fillna("dummy_id")
    predictions_df["score"] = predictions_df["score"].fillna(0)

    #merge each argument with the gold labels
    merged_df = pd.merge(predictions_df, labels_df, how="left", on=["arg_id", "key_point_id"])

    merged_df.loc[merged_df['key_point_id'] == "dummy_id", 'label'] = 0
    merged_df["label_strict"] = merged_df["label"].fillna(0)
    merged_df["label_relaxed"] = merged_df["label"].fillna(1)
    return merged_df


"""
this method chooses the best key point for each argument
and generates a dataframe with the matches and scores
"""
def load_predictions(preds):
    arg =[]
    kp = []
    scores = []
    for arg_id, kps in preds.items():
        best_kp = max(kps.items(), key=lambda x: x[1])
        arg.append(arg_id)
        kp.append(best_kp[0])
        scores.append(best_kp[1])
    return pd.DataFrame({"arg_id" : arg, "key_point_id": kp, "score": scores})


##### END OF METHODS FROM THE KPA EVALUATION FILE ##########

####### OUR METHODS #######

def match_argument_with_keypoints(result, kp_dict, arg_dict):    
        for arg, arg_embedding in arg_dict.items():
            result[arg] = {}
            for kp, kp_embedding in kp_dict.items():
                result[arg][kp] = util.pytorch_cos_sim(arg_embedding, kp_embedding).item()

        return result



def perform_preds(model, arg_df, kp_df, append_topic):
    argument_keypoints = {}
    for topic in arg_df.topic.unique():
        for stance in [-1, 1]:
            topic_keypoints_ids = kp_df[(kp_df.topic==topic) & (kp_df.stance==stance)]['key_point_id'].tolist()
            topic_keypoints = kp_df[(kp_df.topic==topic) & (kp_df.stance==stance)]['key_point'].tolist()
            
            if append_topic:
                topic_keypoints = [topic + ' <SEP> ' + x for x in topic_keypoints]
                
            topic_keypoints_embeddings = model.encode(topic_keypoints, show_progress_bar=False)
            topic_kp_embed = dict(zip(topic_keypoints_ids, topic_keypoints_embeddings))

            topic_arguments_ids = arg_df[(arg_df.topic==topic)&(arg_df.stance==stance)]['arg_id'].tolist()
            topic_arguments = arg_df[(arg_df.topic==topic)&(arg_df.stance==stance)]['argument'].tolist()
            topic_arguments_embeddings = model.encode(topic_arguments, show_progress_bar=False)
            topic_arg_embed= dict(zip(topic_arguments_ids, topic_arguments_embeddings))

            argument_keypoints = match_argument_with_keypoints(argument_keypoints, topic_kp_embed, topic_arg_embed)

    return argument_keypoints


############################

class KeyPointEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example). Checks if distance(sentence,positive_example) < distance(sentence, negative_example).
    """
    def __init__(self, arg_df, kp_df, labels_df, append_topic, main_distance_function: SimilarityFunction = None, name: str = '', batch_size: int = 16, show_progress_bar: bool = False, write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset


        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score

        """
        self.arg_df = arg_df
        self.kp_df = kp_df
        self.labels_df = labels_df
        self.name = name
        self.append_topic=append_topic
        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "triplet_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "mAP_relaxed", "mAP_strict"]
        self.write_csv = write_csv


    @classmethod
    def from_eval_data_path(cls, eval_data_path, subset_name, append_topic, **kwargs):
        arg_df, kp_df, labels_df = load_kpm_data(eval_data_path, subset=subset_name)
        
        return cls(arg_df, kp_df, labels_df, append_topic, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("TripletEvaluator: Evaluating the model on "+self.name+" dataset"+out_txt)

        
        #Perform prediction on the validation/test dataframes
        preds = perform_preds(model, self.arg_df, self.kp_df, self.append_topic)

        merged_df = get_predictions(preds, self.labels_df, self.arg_df)
        
        #Perform evaluation
        mAP_strict, mAP_relaxed = evaluate_predictions(merged_df)
        
        print(f"mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}")
        
        logger.info("mAP strict:   \t{:.2f}".format(mAP_strict*100))
        logger.info("mAP relaxed:   \t{:.2f}".format(mAP_relaxed*100))
        
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, mAP_relaxed, mAP_strict])

            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, mAP_relaxed, mAP_strict])


        return (mAP_strict + mAP_relaxed)/2
