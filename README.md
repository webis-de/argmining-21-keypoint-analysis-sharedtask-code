The provided validation set is split into validation and test sets internally. 

### Trials on our valid

| Experiment      	| mAP Strict         	| mAP Relaxed        	|
|-----------------	|--------------------	|--------------------	|
| Triplet NN      	| 0.844355603203168 	| 0.947050810808719 	|
| Entailment ANLI 	| 0.6893564141187134 	| 0.8970236914678582 	|
| Paraphrase detection 	| 0.7114848640492888 	| 0.878784591933353 	|




### Trials on our test

| Experiment      	| mAP Strict         	| mAP Relaxed        	|
|-----------------	|--------------------	|--------------------	|
| Triplet NN      	| 0.8884606072596714 	| 0.963442996320446 	|
| Entailment ANLI 	| |  	|
| Paraphrase detection 	|  	|  	|


### Meeting notes 18.05
- Split evaluation script into (argument, single keypoint), (argument, multiple keypoints), (argument, no keypoint)
- Data augmentation for paraphrasing and to test other approaches
