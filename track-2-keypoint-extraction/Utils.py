"""
Utils for the approaches
"""
import math
from typing import Tuple, List

# We executed the code on strong CPU clusters without an GPU (ssh compute). Because of this extraordinary
# executing environment, we introduce this flag. To reproduce the results in the paper, enable this flag.
execute_on_ssh_compute = True

import pathlib
import pickle

import loguru
import numpy
from gensim.models import KeyedVectors

import nltk

nltk.download("punkt")
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")
from nltk.corpus import stopwords

setStopWords = set(stopwords.words("english"))
import re

import matplotlib

if execute_on_ssh_compute:
    # see
    # <https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable>
    matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (7.55, 4.0)
plt.rcParams["axes.titlesize"] = "small"
plt.rcParams["axes.titleweight"] = "light"

logger = loguru.logger


# noinspection PyBroadException
class UserLabelCluster:
    classes: []
    classes_to_index_dict: dict

    def __init__(self, user_labels: List[str], word2vec_dict: KeyedVectors, cluster_k: int, word2vec_embedding_size=300,
                 semantic_clustering=True, iteration=0):
        """
        Initiates a UserLabelCluster

        :param user_labels: the user label which will be available during training (do NOT input user labels
            = frames which are in validation or even test data!)
        :param word2vec_dict: the embedding dictionary, for example glove
        :param cluster_k: how many clusters do you want to have? Must be lower equal the number of user labels.
            However, there are some special cases:
            <ul>
                <li> 1: only one cluster, so, there is nothing to predict (making wrong)</li>
                <li> -1: clustering is disabled. It's the vanilla version:
                    <code>
                    set_of_user_frames = {sample.get("frame", "n/a")
                    for sample in samples[: int(len(samples) * training_set_percent * 0.95)]}
                    dict_of_user_frames = {frame: num for num, frame in enumerate(set_of_user_frames)}
                    amount_clusters_dict_of_user_frames = max(dict_of_user_frames.values()) + 1
                    </code>
                </ul>
        :param word2vec_embedding_size: the size of each word2vec embedding
        :param semantic_clustering: determines whether a semantic clustering should be enabled.
            <ul>
                <li><code>True</code>: user labels go through a preprocessing:
                removing stopwords, emphasise keywords, ...</li>
                <li><code>False:</code>: plain vanilla user label is used</li>
            </ul>
        """
        assert cluster_k == -1 or 1 <= cluster_k <= len(user_labels)

        self.classes = list()
        self.classes_to_index_dict = dict()
        self.word2vec_dict = word2vec_dict
        self.word2vec_embedding_size = word2vec_embedding_size
        self.cluster_k = cluster_k

        self.semantic_clustering = semantic_clustering

        if self.cluster_k != 1:
            for user_label in user_labels:
                self.insert_class(user_label=user_label)
        else:
            logger.warning("You choose a special case! "
                           "With only 1 big cluster, all data-points will lay in this cluster. "
                           "Hence, there is nothing to compute...")

        if self.cluster_k > 1:
            path = pathlib.Path("clusters", "{}x{}d_{}_{}c_{}.pkl".format(len(user_labels), word2vec_embedding_size,
                                                                          "semantic" if semantic_clustering else "",
                                                                          cluster_k, iteration))
            path.parent.mkdir(exist_ok=True, parents=True)
            if path.exists():
                logger.debug("You computed the clusters already once, here: {}", path.absolute())
                self.cluster, self.classes_to_index_dict = pickle.load(path.open(mode="rb"))
                logger.success("Successfully loaded the already computed cluster: {}", path.name)
            else:
                logger.trace("Compute the cluster now...")
                self.cluster = nltk.cluster.KMeansClusterer(num_means=self.cluster_k,
                                                            distance=nltk.cluster.cosine_distance,
                                                            repeats=10 + int(math.sqrt(self.cluster_k) * 3),
                                                            avoid_empty_clusters=True)
                self.finalize_class_set()
                logger.success("Yes, we will store it in \"{}\"", path.name)
                pickle.dump((self.cluster, self.classes_to_index_dict), path.open(mode="wb"))
                logger.trace("Pickling done: {}", path.stat())
        elif self.cluster_k == -1:
            logger.warning("You disabled the clustering!"
                           "Hence, it's possible that further predictions will lead to outputs like"
                           "\"Your input is in no particular \"class\"\"")
            self.classes_to_index_dict = {f: i for i, f in enumerate(self.classes)}

    def insert_class(self, user_label: str) -> None:
        """
        Inserts a new user label (frame) which should be used by the cluster

        NOT RECOMMENDED TO USE FROM OUTSIDE!

        :param user_label: the user label which should be inserted
        :return: nothing - just updates the internal structure. Has no effect without using self.finalize_class_set
        """
        logger.debug("Adds \"{}\" to the user label class", user_label)

        if self.cluster_k == -1:
            logger.debug("Clustering is disabled. Hence, just added to the list in set-semantic (current length: {})",
                         len(self.classes))
        final_label_tokens = self.convert_label(user_label=user_label)
        if final_label_tokens not in self.classes:
            self.classes.append(final_label_tokens)
        else:
            logger.debug("\"{}\" was already in the list!", " ".join(final_label_tokens))

    def convert_label(self, user_label: str) -> Tuple[str]:
        """
        FOR INTERNAL USE ONLY!

        :param user_label: the user label (frame)
        :return: a converted tokenized Tuple-list for further processing
        """
        user_label = re.sub(string=user_label, pattern="(?<=\w)\/(?=\w)", repl=" ", count=1)
        final_label_tokens = nltk.word_tokenize(text=user_label, language="english", preserve_line=False)
        for i, token in enumerate(final_label_tokens):
            token = token.lower()
            if token == "v" or token == "v." or token == "vs" or token == "vs.":
                final_label_tokens[i] = "versus"
        if self.semantic_clustering:
            tagged_label = [t_tag for t_tag in nltk.pos_tag(tokens=final_label_tokens, lang="eng", tagset="universal")
                            if t_tag[0] not in setStopWords]
            tagged_label.reverse()
            final_label_tokens = [t_tag[0] for t_tag in tagged_label if t_tag[1] == "NOUN"] + \
                                 [t_tag[0] for t_tag in tagged_label if t_tag[1] == "VERB"] + \
                                 [t_tag[0] for t_tag in tagged_label if t_tag[1] not in ["NOUN", "VERB"]]
        logger.debug("Converted the user label \"{}\" to \"{}\"", user_label, " ".join(final_label_tokens))
        if len(final_label_tokens) > 4:
            logger.warning("The label {} has more than 4 tokens: {}. Discard {}", final_label_tokens,
                           len(final_label_tokens), final_label_tokens[4:])
            final_label_tokens = final_label_tokens[:4]
        elif len(final_label_tokens) == 0:
            logger.warning("We receive an preprocessed user label which is empty!")
            final_label_tokens = ["<pad>"] * 4
        elif len(final_label_tokens) == 1:
            final_label_tokens = final_label_tokens * 4
        elif len(final_label_tokens) == 2:
            final_label_tokens = final_label_tokens * 2
        elif len(final_label_tokens) == 3:
            final_label_tokens.append(final_label_tokens[0])

        return tuple(final_label_tokens)

    def finalize_class_set(self) -> None:
        """
        UPDATES THE INTERNAL STRUCTURE

        :return: nothing
        """
        logger.info("We have {} distinct classes, let's cluster it!", len(self.classes))

        logger.debug("Created a cluster instance {} and this will cluster {} samples", self.cluster, self.classes)
        try:
            assigned_clusters = self.cluster.cluster(vectors=[self.convert_str_list_to_vector(c) for c in self.classes],
                                                     assign_clusters=True, trace=not execute_on_ssh_compute)
        except Exception:
            logger.exception("Failed to cluster the actual class set ({} samples)", len(self.classes))
            return

        self.classes_to_index_dict.clear()
        for i in range(len(self.classes)):
            self.classes_to_index_dict[self.classes[i]] = assigned_clusters[i]

    def convert_str_list_to_vector(self, string_list: Tuple[str]) -> numpy.ndarray:
        """
        FOR INTERNAL USE ONLY!

        :param string_list: a tuple list of tokens. Must be exactly 4
        :return: a one-dimensional (concatenated) numpy-array. See word embeddings
        """
        if len(string_list) != 4:
            logger.error("convert_str_list_to_vector got a too short or long string list: {}. We return a zero-vector!",
                         string_list)
            return numpy.zeros(shape=(self.word2vec_embedding_size +
                                      self.word2vec_embedding_size / 2 +
                                      self.word2vec_embedding_size / 3 +
                                      self.word2vec_embedding_size / 4,),
                               dtype="float32"
                               )
        ret = numpy.zeros(shape=(0,), dtype="float32")
        for i, token in enumerate(string_list):
            logger.trace("Process the {}. token \"{}\"", (i + 1), string_list[i])
            ret = numpy.concatenate([ret,
                                     numpy.average(
                                         numpy.reshape(
                                             self.word2vec_dict[string_list[i]] if string_list[i] in self.word2vec_dict else
                                                                    numpy.negative(
                                                                        numpy.ones(
                                                                            shape=(self.word2vec_embedding_size,),
                                                                            dtype="float32")
                                                                    ),
                                             (int(self.word2vec_embedding_size / (i + 1)), (i + 1))
                                         ),
                                         axis=1)],
                                    axis=0)
        return ret

    def get_index(self, user_label: str) -> int:
        """
        Gets the ground truth one-hot-encoded label for the particular user label

        :param user_label: a user label (frame)
        :type user_label: str
        :return: an numpy array
        """
        final_user_label = self.convert_label(user_label=user_label)
        if self.cluster_k == -1:
            index = self.classes_to_index_dict.get(final_user_label, len(self.classes_to_index_dict))
            return index
        elif self.cluster_k == 1:
            return 0

        if final_user_label in self.classes_to_index_dict.keys():
            cluster_index = self.classes_to_index_dict[final_user_label]
        else:
            logger.info("We never saw the converted user_label \"{}\" - predict the cluster for it!",
                        " ".join(final_user_label))
            cluster_index = self.cluster.classify(vector=self.convert_str_list_to_vector(final_user_label))
            logger.debug("The cluster index of {} is {} - add it to the dictionary!", final_user_label, cluster_index)
            self.classes_to_index_dict[final_user_label] = cluster_index

        return cluster_index

    def get_index_amount(self) -> int:
        """
        :return: the length of a returned vector by self.get_y
        """
        if self.cluster_k == -1:
            return len(self.classes_to_index_dict) + 1

        return self.cluster_k

    def __str__(self) -> str:
        return "{}{}cluster_{}z{}".format("Semantic" if self.semantic_clustering else "",
                                          "-{}d-".format(self.word2vec_embedding_size)
                                          if self.word2vec_embedding_size != 300 else "",
                                          len(self.classes),
                                          self.get_index_amount())