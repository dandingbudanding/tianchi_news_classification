import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


train = pd.read_csv('./train_set.csv', sep='\t')
test = pd.read_csv('./test_a.csv', sep='\t')

train_text = train['text']
test_text = test['text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)

word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

# svd = TruncatedSVD(300)
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)
# X_train = lsa.fit_transform(train_word_features)
X_train = train_word_features
y_train = train['label']

x_train_, x_valid_, y_train_, y_valid_ = train_test_split(train_word_features, y_train, test_size=0.3, shuffle=True, random_state=42)
# X_test = lsa.fit_transform(test_word_features)
X_test = test_word_features


clf = XGBClassifier(learning_rate =0.05,
                    n_estimators=300,
                    max_depth=10,
                    min_child_weight=1,
                    gamma=0.5,
                    reg_alpha=0,               # L1 regularization term on weights.Increasing this value will make model more conservative.
                    reg_lambda=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=1,
                    objective='multi:softmax',
                    num_class=14,
                    nthread=20,
                    seed=1000)
xgb_param = clf.get_xgb_params()
xgTrain = xgb.DMatrix(x_train_, label=y_train_)

print ('Start cross validation')
cvresult = xgb.cv(xgb_param, xgTrain, num_boost_round=500, nfold=5, metrics=['mlogloss'],
     early_stopping_rounds=5, stratified=True, seed=1000)


#交叉验证后最好的树
print('Best number of trees = {}'.format(cvresult.shape[0]))
clf.set_params(n_estimators=cvresult.shape[0])#把clf的参数设置成最好的树对应的参数
print('Fit on the all_trainingsdata')
clf.fit(X_train,y_train,eval_metric=['mlogloss'])

# mode inference
from sklearn.metrics import f1_score

print('Fit on the testingsdata')
print('test f1_score:', f1_score(y_valid_, clf.predict(x_valid_), average='macro'))#测试auc

submission = pd.read_csv('./test_a_sample_submit.csv')
preds = clf.predict(X_test)
submission['label'] = preds
submission.to_csv('./20200729_5zhe_best_alldatafortrain_xgb_submission.csv', index=False)

