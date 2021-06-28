import re
from functools import reduce

import pandas
import torch, bert_score
from pathlib import Path

import gensim.downloader

from Utils import UserLabelCluster

# hyperparameters
max_length_of_keypoint = 100
max_number_of_keypoints = 5
threshold_bert_similarity = .65


if __name__ == '__main__':
    model = gensim.downloader.load("glove-wiki-gigaword-300")

    source_file = Path("..", "data", "test_split_with_aspects.csv")
    df = pandas.read_csv(filepath_or_buffer=str(source_file))

    topics = set(df.get("topic"))
    final_topic_stance_keypoint_dict = dict()

    for topic in topics:
        for stance in [-1,1]:
            all_arguments = [(arg, frozenset([cls.strip(" \"'") for cls in asp.strip("[]").split(",")]))
                             for arg, top, st, asp
                             in zip(df.get("argument"), df.get("topic"), df.get("stance"), df.get("aspects"))
                             if top == topic and st == stance and not isinstance(asp, float)]
            c = UserLabelCluster(user_labels=list(reduce(lambda s1, s2: s1.union(s2), [a[1] for a in all_arguments])),
                                 word2vec_dict=model, cluster_k=max_number_of_keypoints*3, word2vec_embedding_size=300)

            all_arguments_including_cost = [(a[0],
                                             frozenset([c.get_index(asa) for asa in a[1]]),
                                             1+len(a)/float(max_length_of_keypoint) +
                                             (3 if a[0].startswith("I ") else (2 if a[0].startswith("In my opinion")
                                                                               else (.5 if a[0].startswith("\"")
                                                                                     else 0)))) for a in all_arguments
                                            if len(a[0]) < max_length_of_keypoint]
            set_of_keypoints = set()
            covered_aspects = set()

            coverage_list = all_arguments_including_cost
            for i in range(max_number_of_keypoints):
                coverage_list = [(a[0], frozenset(a[1].difference(covered_aspects)), a[2], len(a[1])) for a in coverage_list]
                coverage_list.sort(key=lambda e: (len(e[1]), -e[2]-e[3]), reverse=True)
                add_elem = coverage_list.pop(0)
                set_of_keypoints.add(add_elem)
                covered_aspects.union(add_elem[1])

            candidates = list(set_of_keypoints)
            candidates.sort(key=lambda x: x[2], reverse=False)

            for keypoint_index, keypoint in enumerate(candidates):
                if threshold_bert_similarity < 1:
                    ret = bert_score.score(cands=[keypoint[0]]*len(candidates), refs=[x[0] for x in candidates],
                                           rescale_with_baseline=True, idf=True, lang="en")
                    to_remove = []
                    for cand in range(ret[-1].shape[0]):
                        if cand != keypoint_index and ret[-1][cand] >= threshold_bert_similarity:
                            print("\"{}\" and \"{}\" are similar - discard the second one".format(keypoint[0], candidates[cand][0]))
                            to_remove.append(candidates[cand] if cand > keypoint_index else keypoint)
                    for r in to_remove:
                        if len(candidates) > 3:
                            candidates.remove(r)
                        else:
                            print("\"{}\" can't be removed since there are only 5 keypoints left "
                                  "and it have to be at least 5!", r[0])

            print("Best combination for topic \"{}\"->{} is: {}".format(topic, stance,
                                                                        " ++ ".join(
                                                                            map(lambda x: "{} ({})".format(*x[:2]),
                                                                                candidates)))
                  )

            final_topic_stance_keypoint_dict[(topic, stance)] = \
                [re.sub(string=cand[0].strip(" \""), pattern="^[Ii]n my (personal)? opinion,\s*", repl="")
                 for cand in candidates]


    def mergeDict(dict1, dict2):
        ''' Merge dictionaries and keep values of common keys in list'''
        dict3 = {**dict1, **dict2}
        for key, value in dict3.items():
            if key in dict1 and key in dict2:
                dict3[key] = value + dict1[key] if isinstance(value, list) else [value, dict1[key]]
        return dict3

    df_res = pandas.DataFrame.from_dict(
        data=reduce(lambda x1,x2: mergeDict(x1, x2),
                    [{"key_point_id": [hash(keypoint) for keypoint in v], "key_point": v, "topic": [k[0]]*len(v),
                      "stance": [k[1]]*len(v)} for k, v in final_topic_stance_keypoint_dict.items()]),
        orient="columns")

    df_res.to_csv(path_or_buf="{}-key_points-x{}.csv".format(source_file.stem, max_number_of_keypoints),
                  encoding="utf-8", errors="replace", index=False)