# Pipeline

## Codes

* TODO: test.py
* facealigner.py: a stand-alone module. Used to transform original image into face & points

* feature_utils.py: a stand-alone module. Used to extract features and dump them in pkl file.
* make_dataset.py: a stand-alone module. Used to make n-fold dataset pkl file using extracted pkl features
* train.py, it calls:
  * classifier.py: classifiers, including LogisticRegression & SVM; **Now implemented by sklearn**. Replaced by ours later. 
  * compressor.py: compressors, including PCA only. We may use SparsePCA later. **Now implemented by sklearn**. Replaced by ours later.



