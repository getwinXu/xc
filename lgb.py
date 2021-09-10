import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy
import gc
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import roc_auc_score
def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df




def to_str(x):
    if isinstance(x,str):
        xlist = x.split(':')
        return xlist[1]
    else:
        return x

class Config(object):
    def __init__(self,data_dir):
        self.data_path = data_dir + 'train.txt'
        self.test_path = data_dir + 'test.txt'
        self.vec=16



def get_w2v_dire(sentence,config):
    model=Word2Vec(sentence,vector_size=config.vec,min_count=1,sg=1)
    dict={}
    for i in model.wv.index_to_key:
        dict[i]=model.wv[i]
    # file=open('./88dict.pkl','wb')
    # pickle.dump(dict,file)
    return dict

def w2v(word,dict,config,iscata = False):
    if word in dict:
        return dict[word]
    elif iscata:
        return [0]*config.vec
    else:
        return
    #     print('————————————————————————————————————————————错了铁罕汗————————————————————————————————————————————————————————————')
def getmeanVec(st,dict,config):   #函数作用是将字符串变成均值向量,返回numpy类型的向量
    art=[w2v(i,dict,config) for i in st.split(',')]
    a=np.array(art)
    return np.mean(a,axis=0)

# config = Config('/workspace/mdata/')
config = Config('D:\\浏览器下载\\')
n_components = 12
t = time.time()
path = config.data_path
tmpdf = pd.read_csv(path, sep='\t', header=None,nrows = 10000)
tmpdf.columns = ['click', 'resume'] + [1,2,3,5,6,7,13,16,18,20,21,22,23,24,25,26,27,28,29,30,
                                       31,32,33,34,35,36,37,38,39,40,41,43,45,46,47,48,52,
                                       54,55,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,
                                       73,74,75,76,77,78,79,80,81,82,83,84,87,88,89]
# multivalue_train=tmpdf[[7, 47, 48, 88, 89]]
tmpdf.drop([5, 6], axis=1, inplace=True)

test_df = pd.read_csv(config.test_path, sep='\t', header=None,nrows = 10000)
test_df.columns=[1,2,3,7,13,16,18,20,21,22,23,24,25,26,27,28,29,30,
                31,32,33,34,35,36,37,38,39,40,41,43,45,46,47,48,52,
                54,55,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,
                73,74,75,76,77,78,79,80,81,82,83,84,87,88,89]
# multivalue_test=test_df[[7, 47, 48, 88, 89]]
# multivalue=multivalue_train.append(multivalue_test,ignore_index=True)
# multivalue=multivalue.applymap(to_str)

tuijian=tmpdf.append(test_df,ignore_index=True)
del test_df,tmpdf
gc.collect()
tuijian=tuijian.applymap(to_str)                        #所有特征
print('加载完毕,开始对类别特征编码',time.time() - t)
# t = time.time()

mulfea89 = tuijian[89].apply(lambda x: [_.split(';') for _ in x.split(',')])
mulfea88 = tuijian[88].apply(lambda x: x.split(','))

#对88的id进行处理
# tuijian[88] =tuijian[88].apply(lambda x:[ i for i in x.split(',')])
# sentence = tuijian.apply(lambda x:[x[16]] if x['click'] == 1 else [],axis = 1)
# tuijian[88] +=sentence
# idDire=get_w2v_dire(tuijian[88],config)
# fea88vec=tuijian[88].apply(lambda x:getmeanVec(x,idDire,config))   #作用是获取用户点击的平均w2v向量,getmeanVec返回值是平均向量的numpy
#
# idfeavec=tuijian[16][tuijian[16] != '-1'].apply(lambda x:w2v(x,idDire,config))
# distance=pd.Series([0]*len(tuijian))
# for i in range(config.vec):
#     tuijian['UIDVec{}'.format(i)]=fea88vec.apply(lambda x:x[i])
# tuijian['IdVec{}'.format(i)]=idfeavec.apply(lambda x:x[i])
# distance=distance+(tuijian['UIDVec{}'.format(i)]-tuijian['IdVec{}'.format(i)]).apply(np.square)
# tuijian['IdVecDis']=distance
# tuijian.drop([16],axis=1,inplace=True)
# 对88的id进行处理
sentence = list(mulfea88.values)
idDire = get_w2v_dire(sentence, config)
idfeavec = tuijian[16].apply(lambda x: w2v(x, idDire,config))
exis_se = idfeavec[ ~ idfeavec.isnull()]
exis_ind = exis_se.index
no_ind = idfeavec[idfeavec.isnull()].index
ex_id_df = tuijian.loc[exis_ind,[43,45]]
no_id_df = tuijian.loc[no_ind,[43,45]]
fea = []
for i in range(config.vec):
    ex_id_df[f'id_vec_{i}'] =exis_se.apply(lambda x:x[i])
    fea.append(f'id_vec_{i}')
no_id_df = pd.merge(no_id_df,ex_id_df.groupby([43,45]).mean(),left_on = [43,45],right_index=True,how = 'left').fillna(0)
id_vec_df = ex_id_df.append(no_id_df)
tuijian = tuijian.merge(id_vec_df[fea],left_index=True,right_index=True,how = 'left')
fea88vec = tuijian[88].apply(lambda x: getmeanVec(x, idDire, config))
for i in range(config.vec):
    tuijian['UIDVec{}'.format(i)]=fea88vec.apply(lambda x:x[i])

tuijian = reduce_mem(tuijian)
del idDire,idfeavec,exis_se,ex_id_df,no_id_df,fea88vec
gc.collect()

print('结束16merge：',time.time() - t)











click_user = []
for i in range(len(mulfea88)):
    if mulfea88[i] != ['']:
        for j in range(len(mulfea88[i])):
            click_user.append([i, mulfea88[i][j]] + mulfea89[i][j])
click_df2 = pd.DataFrame(click_user)
click_df2.columns = ['index',16,20,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,41,43,45,46,47,90,91,92]
click_df2 = reduce_mem(click_df2)


#对7特征进行处理
x = list(tuijian[7].apply(lambda x:x.replace(',',' ')))
tfv = TfidfVectorizer(ngram_range=(1,1))
tfv.fit(x)
x_tfidf = tfv.transform(x)
svd = TruncatedSVD(n_components=n_components)
x_svd = svd.fit_transform(x_tfidf)
for i in range(n_components):
    tuijian[f'7_tfidf_svd_{i}'] = x_svd[:,i]
# modelpca = PCA(n_components=n_components)
# x_pca = modelpca.fit_transform(x_tfidf)
del tfv,x_tfidf,svd,x_svd
gc.collect()
tuijian = reduce_mem(tuijian)

#对47职业标题标签进行编码
x = list(tuijian[47].apply(lambda x:x.replace(',',' ')))
x2 = list(click_df2[47].apply(lambda x:x.replace(',',' ')))
x_all =x+x2
tfv = TfidfVectorizer(ngram_range=(1, 1))
tfv.fit(x_all)
x_tuijian = tfv.transform(x)
x_click = tfv.transform(x2)

svd = TruncatedSVD(n_components=n_components)
x_svd_tuijian = svd.fit_transform(x_tuijian)
x_svd_click = svd.fit_transform(x_click)
fea_list = ['index']
for i in range(n_components):
    tuijian[f'47_tfidf_svd_{i}'] = x_svd_tuijian[:, i]
    click_df2[f'47_tfidf_use_{i}'] = x_svd_click[:, i]
    fea_list.append(f'47_tfidf_use_{i}')
tuijian = tuijian.merge(click_df2[fea_list].groupby('index').mean(),left_index=True,right_index=True,how='left')
fea_list.pop(0)
tuijian[fea_list] = tuijian[fea_list].fillna(0)
del x,x2,tfv,x_tuijian,x_click,x_svd_click,x_svd_tuijian
gc.collect()
tuijian = reduce_mem(tuijian)

#对48职业内容标签进行编码
x = list(tuijian[48].apply(lambda x: str([_.split('_')[0] for _ in x.split(',')]).lstrip('[').rstrip(']').replace(',',' ')))
tfv = TfidfVectorizer(ngram_range=(1,1))
tfv.fit(x)
x_tfidf = tfv.transform(x)
svd = TruncatedSVD(n_components=n_components)
x_svd = svd.fit_transform(x_tfidf)
for i in range(n_components):
    tuijian[f'48_tfidf_svd_{i}'] = x_svd[:,i]
# modelpca = PCA(n_components=n_components)
# x_pca = modelpca.fit_transform(x_tfidf)
del tfv,x_tfidf,svd,x_svd,click_df2
gc.collect()
tuijian = reduce_mem(tuijian)

#对89用户浏览记录特征处理
vecfea = [43,45,46,47]
fea_ind = [17,18,19,20]
for i,j in zip(vecfea,fea_ind):
    sentence = mulfea89.apply(lambda x:[_[j] for _ in x] if x != [['']] else [''])
    idDire = get_w2v_dire(sentence, config)
    feavec = tuijian[i].apply(lambda x: w2v(x, idDire,config,True))
    usevec =sentence.apply(lambda x: np.array([w2v(_,idDire,config) for _ in x]).mean(axis = 0))
    for fea in range(config.vec):
        tuijian[f'{i}_vec_{fea}'] = feavec.apply(lambda x:x[fea])
        tuijian[f'{i}_mean_{fea}'] = usevec.apply(lambda x:x[fea])
del feavec,usevec
gc.collect()
tuijian = reduce_mem(tuijian)

fea_ind = [21,22,23]
for j in fea_ind:
    sentence = mulfea89.apply(lambda x: [_[j] for _ in x] if x != [['']] else [''])
    idDire = get_w2v_dire(sentence, config)
    usevec = sentence.apply(lambda x: np.array([w2v(_, idDire, config) for _ in x]).mean(axis=0))
    for fea in range(config.vec):
        tuijian[f'{i}_mean_{fea}'] = usevec.apply(lambda x: x[fea])
tuijian = reduce_mem(tuijian)
del sentence,idDire,usevec
gc.collect()

for i in range(17):
    click_fea = mulfea89.apply(lambda x: [float(_[i]) for _ in x] if x != [['']] else [-1])
    tuijian[f'fea{i}_mean'] = click_fea.apply(lambda x: sum(x)/len(x))
tuijian = reduce_mem(tuijian)

# # 获取类别编码的个数
catefea = [43, 45, 46, 52]
# for i in catefea:
#     lbc = LabelEncoder()
#     tuijian[i] = lbc.fit_transform(tuijian[i])
#     tuijian[f'fea{i}_counter'] = tuijian[i].map(tuijian[i].value_counts())

tuijian.drop([13,16,7,47,48,88,89],axis = 1,inplace = True)
tuijian = tuijian.astype('float')
tuijian = reduce_mem(tuijian)
print('结束特征构建：',time.time() - t)




del tuijian['resume']
traindata = tuijian[~ tuijian['click'].isnull()]
label = traindata['click']
del traindata['click']
testdata = tuijian[tuijian['click'].isnull()]
testdata = testdata.reset_index(drop=True)
del testdata['click'],tuijian
gc.collect()

n_fold = 5
kfd = KFold(n_fold,shuffle=True,random_state=2105)
predict = np.zeros(len(testdata))
val_predict = np.zeros(len(traindata))
for i,(train_ind,val_ind) in enumerate(kfd.split(traindata)):
    train_x = traindata.iloc[train_ind]
    train_y = label[train_ind]
    val_x = traindata.iloc[val_ind]
    val_y = label[val_ind]
    clf = LGBMRegressor(learning_rate=0.01,
                         objective='regression',
                        n_estimators=200,
                        num_leaves=255,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=2019,
                        metric=None
                        )
    print(f'————————————————————————————————————第{i}次：train——————————————————————————————————————————————————————')
    print(time.time() - t)
    clf.fit(
        train_x, train_y,
        eval_set=[(val_x, val_y)],
        eval_metric='auc',
        categorical_feature=catefea,
        early_stopping_rounds=200,
        verbose=50
    )
    print('runtime:', time.time() - t)
    print(f'————————————————————————————————————第{i}次：predict——————————————————————————————————————————————————————')
    test_y = clf.predict(testdata)
    predict += test_y/n_fold
    val_predict[val_ind] = clf.predict(val_x)
    print(f'第{i}次验证auc：',roc_auc_score(val_y,val_predict[val_ind]))
auc = roc_auc_score(label,val_predict)
print('预测点击auc：',auc)
pd.DataFrame(predict).to_csv("/workspace/model/submission.csv", header=False, index=False)
# pd.DataFrame(predict).to_csv("./submission.csv", header=False, index=False)



