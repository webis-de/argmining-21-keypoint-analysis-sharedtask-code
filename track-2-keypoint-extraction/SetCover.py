import re
from functools import reduce

import pandas
import torch, bert_score

import gensim.downloader

from Utils import UserLabelCluster

if __name__ == '__main__':
    model = gensim.downloader.load("glove-wiki-gigaword-300")

    df = pandas.read_csv(filepath_or_buffer="../data/sample_aspects.csv")

    topics = set(df.get("topic"))
    final_topic_stance_keypoint_dict = dict()

    for topic in topics:
        for stance in [-1,1]:
            all_arguments = [(arg, frozenset([cls.strip(" \"'") for cls in asp.strip("[]").split(",")]))
                             for arg, top, st, asp
                             in zip(df.get("argument"), df.get("topic"), df.get("stance"), df.get("aspects"))
                             if top == topic and st == stance and not isinstance(asp, float)]
            c = UserLabelCluster(user_labels=list(reduce(lambda s1, s2: s1.union(s2), [a[1] for a in all_arguments])),
                                 word2vec_dict=model, cluster_k=30, word2vec_embedding_size=300)

            all_arguments_including_cost = [(a[0],
                                             frozenset([c.get_index(asa) for asa in a[1]]),
                                             1+len(a)/100. +
                                             (3 if a[0].startswith("I ") else (2 if a[0].startswith("In my opinion")
                                                                               else (.5 if a[0].startswith("\"")
                                                                                     else 0)))) for a in all_arguments
                                            if len(a[0]) < 100]
            set_of_keypoints = set()
            covered_aspects = set()

            coverage_list = all_arguments_including_cost
            for i in range(10):
                coverage_list = [(a[0], frozenset(a[1].difference(covered_aspects)), a[2], len(a[1])) for a in coverage_list]
                coverage_list.sort(key=lambda e: (len(e[1]), -e[2]-e[3]), reverse=True)
                add_elem = coverage_list.pop(0)
                set_of_keypoints.add(add_elem)
                covered_aspects.union(add_elem[1])

            candidates = list(set_of_keypoints)
            candidates.sort(key=lambda x: x[2], reverse=False)

            for keypoint_index, keypoint in enumerate(candidates):
                ret = bert_score.score(cands=[keypoint[0]]*len(candidates), refs=[x[0] for x in candidates],
                                       rescale_with_baseline=True, idf=True, lang="en")
                to_remove = []
                for cand in range(ret[-1].shape[0]):
                    if cand != keypoint_index and  ret[-1][cand] >= .6:
                        print("\"{}\" and \"{}\" are similar - discard the second one".format(keypoint[0], candidates[cand][0]))
                        to_remove.append(candidates[cand] if cand > keypoint_index else keypoint)
                for r in to_remove:
                    candidates.remove(r)

            print("Best combination for topic \"{}\"->{} is: {}".format(topic, stance,
                                                                        " ++ ".join(map(lambda x: "{} ({})".format(*x[:2],), candidates))))

            final_topic_stance_keypoint_dict[(topic, stance)] = \
                [re.sub(string=cand[0], pattern="^[Ii]n my (personal)? opinion,\s*", repl="") for cand in candidates]