NAME: IBM Debater(R) - ArgKP

VERSION: v1

RELEASE DATE: May 10, 2020

DATASET OVERVIEW

24,093 (argument, key point) pairs labeled as matching/non-matching, for 28 controversial topics. 
For each pair, the topic and stance are also indicated.

The dataset is released under the following licensing and copyright terms:
• (c) Copyright IBM 2020. Released under Community Data License Agreement – Sharing, Version 1.0 (https://cdla.io/sharing-1-0/).

The dataset is described in the following publication: 

• From Arguments to Key Points: Towards Automatic Argument Summarization. Roy Bar-Haim, Lilach Eden, Roni Friedman, 
Yoav Kantor, Dan Lahav and Noam Slonim. ACL 2020.

Please cite this paper if you use the dataset.

CONTENTS

The CSV file, ArgKP_dataset.csv, contains the following columns for each (argument, key point) pair:
1. topic
2. argument
3. key_point	
4. stance: 1 (pro) / -1 (con)
5. label: 1 (matching)/ 0 (non-matching)
