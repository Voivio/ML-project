# Pipeline

## Results

MB LBP on test split 9

| feature scales                      | n       | agg           | norm | classifier          | train acc f1 | test acc f1  |
| ----------------------------------- | ------- | ------------- | ---- | ------------------- | ------------ | ------------ |
| 3, 9, 17, 25, 49 (data/tenfold_lfw) | 32      | minus-abs     | no   | svm                 | 98.89, 98.89 | 73.50, 71.56 |
|                                     | 64      |               |      |                     | 98.61, 98.62 | 77.17, 76.90 |
|                                     | 128     |               |      |                     | 97.61, 96.61 | 77.00, 76.92 |
|                                     | **256** |               |      |                     | 95.98, 95.96 | 78.17, 78.70 |
|                                     | 512     |               |      |                     | 93.41, 93.34 | 77.33, 78.27 |
|                                     | 1024    |               |      |                     | 90.54, 90.40 | 78.17, 78.06 |
|                                     | 32      |               |      | logistic regression | 74.46, 74.34 | 74.33, 74.25 |
|                                     | 64      |               |      |                     | 77.78, 77.64 | 76.83, 76.72 |
|                                     | 128     |               |      |                     | 79.39, 79.26 | 76.67, 76.43 |
|                                     | 256     |               |      |                     | 81.70, 81.70 | 79.17, 79.13 |
|                                     | 512     |               |      |                     | 84.26, 84.24 | 76.76, 76.19 |
|                                     | 1024    |               |      |                     | 88.63, 88.65 | 76.33, 75.68 |
|                                     | 256     | mul_minus-abs |      | svm                 | 99.63, 99.63 | 68.67, 73.37 |
|                                     | 512     |               |      |                     | 99.41, 99.41 | 65.67, 73.86 |
|                                     | 2048    |               |      |                     | x            | x            |
|                                     | 512     |               | yes  |                     | x            | 58.83, 56.89 |
|                                     | 256     | mul           |      |                     | 94.20, 94.16 | 73.33, 69.92 |

## Codes

* TODO: test.py
* facealigner.py: a stand-alone module. Used to transform original image into face & points

* feature_utils.py: a stand-alone module. Used to extract features and dump them in pkl file.
* make_dataset.py: a stand-alone module. Used to make n-fold dataset pkl file using extracted pkl features
* train.py, it calls:
  * classifier.py: classifiers, including LogisticRegression & SVM; **Now implemented by sklearn**. Replaced by ours later. 
  * compressor.py: compressors, including PCA only. We may use SparsePCA later. **Now implemented by sklearn**. Replaced by ours later.



