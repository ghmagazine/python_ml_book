
# coding: utf-8

# ### 2.2.1 データの読み込み

# In[1]:


import pandas as pd
iris = pd.read_csv('iris.data.csv', header=None)


# In[2]:


iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']


# In[3]:


pd.read_csv('iris.data.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])


# In[4]:


iris.info()


# ### 2.2.2 データへのアクセス

# In[5]:


iris.iloc[0, 0]  # 1行目で1列目のデータ


# In[6]:


iris.iloc[0, 1]  # 1行目で2列目のデータ


# In[7]:


iris.loc[:, ['sepal_length', 'sepal_width']]  # 複数の列名を指定


# In[8]:


iris.loc[:, ['sepal_length', 'sepal_width']].iloc[1:10, :]


# In[9]:


iris.head(n=5)  # データフレームの先頭から5行


# In[10]:


iris.tail(n=5)  # 末尾のデータを表示


# ### 2.2.3 1次元データ：Series

# In[11]:


type(iris.iloc[:, 0])


# ### 2.2.5 事例：型の変換

# In[12]:


iris['class'] = iris['class'].astype('category')


# In[13]:


iris.info()


# In[14]:


iris = pd.read_csv('iris.data.csv',
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'],
                   dtype={'sepal_length': 'float64',
                          'sepal_width': 'float64',
                          'petal_length': 'float64',
                          'petal_width': 'float64',
                          'class': 'category'})


# In[15]:


iris['sepal_length'].astype('int64')


# In[16]:


iris['sepal_length'].astype('object')


# In[17]:


# datetime64への変換
datetime_sample1 = pd.Series(['2018-10-01 00:00', '2018-10-02 00:00', '2018-10-03 00:00'])
datetime_sample1 = pd.to_datetime(datetime_sample1)


# In[18]:


datetime_sample2 = pd.Series(['2018年10月01日 00時00分', '2018年10月03日 00時00分', '2018年10月05日 00時00分'])
datetime_sample2 = pd.to_datetime(datetime_sample2, format='%Y年%m月%d日 %H時%M分')


# In[19]:


datetime_sample2 - datetime_sample1  # timedelta型


# In[20]:


# データを読み込むときに日付データとして処理
test_datetime_csv = '''
x,y
10,2018-10-01 00:00
20,2018-10-02 00:00
'''
with open('test_datetime1.csv', 'w') as f:
    f.write(test_datetime_csv)
pd.read_csv('test_datetime1.csv', parse_dates=['y']).info()


# In[21]:


raw_csv = '''
x,y
10,2018年10月01日 00時00分
20,2018年10月05日 00時00分
'''
with open('test_datetime2.csv', 'w') as f:
    f.write(raw_csv)
def parser(date):
    return pd.datetime.strptime(date, '%Y年%m月%d日 %H時%M分')
pd.read_csv('test_datetime2.csv', parse_dates=['y'], date_parser=parser).info()


# ### 2.3.1 データフレームの作り方

# In[22]:


sample_data = {'x1': [1,2,3], 'x2': [10, 20, 30]}
pd.DataFrame(sample_data)


# In[23]:


import numpy as np  # Numpyの読み込み
matrix = np.random.randn(10,3)  # ランダムな数字が入っているNumpyの行列
pd.DataFrame(matrix)  # Numpyの行列をデータフレームに変換


# ### 2.3.3 データフレームをリストや辞書型に変換

# In[24]:


iris['class'].tolist()  # 配列を取得する


# In[25]:


iris.to_dict()  # keyが列名、valueに行名がkeyでデータがvalueな辞書を入れ子に持つ辞書


# In[26]:


iris.to_dict('records')  # keyが列名、valueがデータとなる辞書をレコード単位で配列にする


# ### 2.4.2 ランキング

# In[27]:


iris['sepal_length'].rank()


# ### 2.4.3 データの並び替え

# In[28]:


# sort_valuesで並び替え
iris.sort_values(by=['sepal_length'])


# In[29]:


# sort_indexで並び替え
iris.sort_index()


# ### 2.4.4 基本的な集計

# In[30]:


iris['sepal_length'].sum()  # sepal_lengthの合計値


# In[31]:


iris['sepal_width'].sum()  # sepal_widthの合計値


# In[32]:


iris['sepal_length'].mean()  # sepal_lengthの平均値


# In[33]:


iris['sepal_length'].max()  # sepal_lengthの最大値


# In[34]:


iris['sepal_length'].describe()


# ### 2.4.5 グループごとの集計

# In[35]:


iris.groupby('class').sum()


# ### 2.4.6 複数の集計を計算

# In[36]:


iris.groupby('class').aggregate([np.mean, np.sum])


# In[37]:


iris.groupby('class').aggregate([np.mean, np.sum])['sepal_length']['mean']


# ### 2.5.1 TSV形式のデータ

# In[38]:


iris.to_csv('iris.data.tsv', sep='\t', index=None, header=None)


# In[39]:


pd.read_table('iris.data.tsv', header=None)


# ### 2.5.2 Excel形式のデータ

# In[40]:


iris.to_excel('iris.data.xlsx', index=None)


# In[41]:


pd.read_excel('iris.data.xlsx', sheet_name=0, index=None)


# ### 2.5.3 htmlのテーブルを読む

# In[42]:


import requests
res = requests.get('http://gihyo.jp/site/profile')
pd.read_html(res.text)


# ### 2.5.4 メモリに乗らないデータを逐次的に読み込む

# In[43]:


for df in pd.read_csv('iris.data.csv', chunksize=10):
    print(df.shape)


# ### 2.6.2 SQLite形式のデータを作る

# In[44]:


import sqlite3
con = sqlite3.connect('iris.data.db')
iris.to_sql('iris', con,  if_exists='replace', index=None)


# ### 2.6.3 SQLの実行

# In[45]:


pd.read_sql('select * from iris', con)


# In[46]:


pd.read_sql('select * from iris where class="Iris-setosa"', con)


# ### 2.7.1 ライブラリとデータの読み込み

# In[47]:


import pandas as pd
df = pd.read_csv('survey_results_public.csv', low_memory=False)
df.info()


# In[48]:


for column in df.columns:
    print(column)


# In[49]:


df['Respondent'].nunique()  # 回答番号がすべて異なっていることを確認


# In[50]:


df['Hobby'].unique()


# In[51]:


df['Hobby'].value_counts()


# ### 2.7.2 カテゴリカルな列を特定

# In[52]:


import numpy as np

category_columns = [
 column for column in df.columns if df[column].nunique() < 100 and df[column].dtype == np.dtype('O')
]
print(len(category_columns))


# In[53]:


for column in category_columns:
    df[column] = df[column].astype('category')


# In[54]:


df.info()  # 型の変換が出来ているかを確認


# In[55]:


# 列名と型の対応を作る
import json
data_dtypes = {column:str(df[column].dtype) for column in df.columns}
json.dump(data_dtypes, open('data_dtypes.json', 'w'))


# In[56]:


df = pd.read_csv('survey_results_public.csv', dtype=json.load(open('data_dtypes.json')))


# ### 2.7.3 データの整形1 - 複数回答を異なる列へ展開

# In[57]:


df['LanguageWorkedWith'].head()


# In[58]:


language_worked_with = df.LanguageWorkedWith.str.split(';',expand=True)


# In[59]:


languages = set()
for values in language_worked_with.values:
    for value in values:
        if isinstance(value, str):
            languages.add(value)
languages = sorted(list(languages))


# In[60]:


def language_dummy_variable(languages):
    def _(x):
        x = set(x.tolist())
        return pd.Series([
            True if language in x else False
            for language in languages
        ])
    return _

sorted_language_worked_with = language_worked_with.apply(language_dummy_variable(languages), axis=1)
sorted_language_worked_with.columns = ['languageWorkedWith_' + language for language in languages]


# In[61]:


sorted_language_worked_with


# In[62]:


np.where(sorted_language_worked_with.sum().sort_values(ascending=False).index == 'languageWorkedWith_Python')


# In[63]:


# データフレームの結合
df = pd.concat([df, sorted_language_worked_with], axis=1)


# In[64]:


df.info()


# ### 2.7.4 データの整形2 - 4種類の回答を3種類にまとめる

# In[65]:


def student_yes_or_no(x):
    if isinstance(x, float):
        return 'Unknown'
    elif x == 'No':
        return 'No'
    else:
        return 'Yes'


# In[66]:


# 様々なデータ加工方法


# In[67]:


get_ipython().run_cell_magic('timeit', '', "df['Student_1'] = df.Student.map(student_yes_or_no)")


# In[68]:


get_ipython().run_cell_magic('timeit', '', "df['Student_1'] = [student_yes_or_no(student) for student in df.Student]")


# In[69]:


get_ipython().run_cell_magic('timeit', '', "df['Student_1'] = df[['Student']].applymap(student_yes_or_no)")


# In[70]:


get_ipython().run_cell_magic('timeit', '', "student_status = pd.DataFrame({'Student': ['No', 'Yes, part-time', 'Yes, full-time', float('nan')], 'Student_1': ['No', 'Yes', 'Yes', 'Unknown']})\nmerged_df = pd.merge(df, student_status, on='Student', how='left')")


# ### 2.7.5 データの整形3 - 条件に一致する行を抽出

# In[71]:


df.query("Hobby == 'Yes'")


# ### 2.7.6 データの整形4 - 縦方向のデータを横方向のデータに変換

# In[72]:


df[['Hobby', 'YearsCoding']].    groupby(['Hobby', 'YearsCoding']).    size().    reset_index(name='counts')


# In[73]:


hobby_years_coding_count = df[['Hobby', 'YearsCoding']].    groupby(['Hobby', 'YearsCoding']).    size().    reset_index(name='counts').    pivot(index='YearsCoding', columns='Hobby', values='counts')
hobby_years_coding_count


# In[74]:


hobby_years_coding_count.index


# In[75]:


hobby_years_coding_count.index = hobby_years_coding_count.index.set_categories(['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years',
          '15-17 years', '18-20 years', '21-23 years',
          '24-26 years', '27-29 years', '30 or more years'
          ])
hobby_years_coding_count.sort_index(inplace=True)


# In[76]:


hobby_years_coding_count


# ### 2.7.7 Plotlyによる可視化

# In[77]:


import plotly
# JupyterLabでplotlyによるグラフをオフライン表示するための初期設定
plotly.offline.init_notebook_mode()


# In[78]:


plotly_data = [
    plotly.graph_objs.Bar(
        x=hobby_years_coding_count.index,
        y=hobby_years_coding_count['Yes'] / sum(hobby_years_coding_count['Yes']),
    name='Yes'),
    plotly.graph_objs.Bar(
        x=hobby_years_coding_count.index,
        y=hobby_years_coding_count['No'] / sum(hobby_years_coding_count['No']),
    name='No'),
]
layout = plotly.graph_objs.Layout(width=700, height=400)
fig = plotly.graph_objs.Figure(data=plotly_data, layout=layout)
plotly.offline.plot(fig)  # 図2.7 趣味としてプログラミングするか否かによるコーディング経験の比較


# ### 2.7.8 1連の前処理を連続して記述するメソッドチェーン

# In[79]:


from functools import reduce

# 2.7.3 データの整形1 - 複数回答を異なる列へ展開 で解説
def to_wide_language_worked_with(df):
    language_worked_with = df.LanguageWorkedWith.str.split(';',expand=True)

    languages = set()
    for values in language_worked_with.values:
        for value in values:
            if isinstance(value, str):
                languages.add(value)
    languages = sorted(list(languages))

    sorted_language_worked_with = language_worked_with.apply(language_dummy_variable(languages), axis=1)
    sorted_language_worked_with.columns = ['languageWorkedWith_' + language for language in languages]

    return pd.concat([df, sorted_language_worked_with], axis=1)

# 2.7.2 カテゴリカルな列を特定 で解説
def object2category(df):
    category_columns = [
     column for column in df.columns if df[column].nunique() < 100 and df[column].dtype == np.dtype('O')
    ]
    for column in category_columns:
        df[column] = df[column].astype('category')
    return df

# pipeやassignによるメソッドチェーンを用いた処理方法
df = pd.read_csv('survey_results_public.csv', low_memory=False).    query("Hobby == 'Yes'").    pipe(object2category).    pipe(to_wide_language_worked_with).    assign(Student_1=lambda x: x.Student.map(student_yes_or_no))  # student_yer_or_noは2.7.4で紹介しました


# ### 2.7.9 正規化と正則化

# In[80]:


from sklearn.preprocessing import StandardScaler
float_columns = [name for ftype, name in zip(df.ftypes, df.ftypes.index) if ftype.find('float64') == 0]
df[float_columns]  # 変換前


# In[81]:


sc = StandardScaler()
sc.fit(df[float_columns])
df_std = sc.transform(df[float_columns])
df[float_columns] = df_std
df[float_columns]  # 変換後


# ### 2.7.10 外れ値

# In[82]:


df = pd.read_csv('survey_results_public.csv', dtype=json.load(open('data_dtypes.json')))
plotly_data = [
    plotly.graph_objs.Histogram(
        x=df.ConvertedSalary
    ),
]
layout = plotly.graph_objs.Layout(width=700, height=400)
fig = plotly.graph_objs.Figure(data=plotly_data, layout=layout)
plotly.offline.plot(fig)  # 図2.8 ConvertedSalaryに外れ値が含まていることを確認できるヒストグラム


# In[83]:


plotly_data = [
    plotly.graph_objs.Histogram(
        x=np.log10(df.ConvertedSalary)
    ),
]
layout = plotly.graph_objs.Layout(width=700, height=400)
fig = plotly.graph_objs.Figure(data=plotly_data, layout=layout)
plotly.offline.plot(fig)  # 図2.9 対数変換したConvertedSalary


# In[84]:


anomaly_std = np.std(df.ConvertedSalary[df.ConvertedSalary > 0]) * 3
converted_salary_mean = np.mean(df.ConvertedSalary[df.ConvertedSalary > 0])
anomaly_threshold = converted_salary_mean + anomaly_std
anomaly_flags = df['ConvertedSalary'] > anomaly_threshold
df[anomaly_flags]['ConvertedSalary'].shape


# In[85]:


# 閾値以上の年収の人を外れ値とする
df['anomaly_salary'] = df['ConvertedSalary'] > anomaly_threshold

# 2.7.6 データの整形4 - 縦方向のデータを横方向のデータに変換 で解説
anomaly_salary_years_coding = df[['anomaly_salary', 'YearsCoding']].    groupby(['anomaly_salary', 'YearsCoding']).    size().    reset_index(name='counts').    pivot(index='YearsCoding', columns='anomaly_salary', values='counts')

# 2.7.6 データの整形4 - 縦方向のデータを横方向のデータに変換 で解説
anomaly_salary_years_coding.index = anomaly_salary_years_coding.index.set_categories(['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years',
          '15-17 years', '18-20 years', '21-23 years',
          '24-26 years', '27-29 years', '30 or more years'
          ])
anomaly_salary_years_coding.sort_index(inplace=True)
anomaly_salary_years_coding.columns


# In[86]:


# 2.7.7 Plotlyによる可視化 で解説
plotly_data = [
    plotly.graph_objs.Bar(
        x=anomaly_salary_years_coding.index,
        y=anomaly_salary_years_coding[True] / sum(anomaly_salary_years_coding[True]),
    name='True'),
    plotly.graph_objs.Bar(
        x=anomaly_salary_years_coding.index,
        y=anomaly_salary_years_coding[False] / sum(anomaly_salary_years_coding[False]),
    name='False'),
]
layout = plotly.graph_objs.Layout(width=700, height=400)
fig = plotly.graph_objs.Figure(data=plotly_data, layout=layout)
plotly.offline.plot(fig)  # 図2.10 コーディングの経験年数の棒グラフを高年収とそれ以外の人で描画


# ### 2.7.11 データのサンプリング

# In[87]:


# サンプルサイズを指定する方法
df_sample = df.sample(n=10, random_state=1)


# In[88]:


# サンプリング割合を指定する方法
df_sample = df.sample(frac=0.1, random_state=1)


# In[89]:


# scikit-learnによるサンプリング
import sklearn.model_selection
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, df.ConvertedSalary, test_size=0.33, random_state=1)


# ### 2.7.13 欠損値の補完

# In[90]:


df.ConvertedSalary.fillna(np.nanmean(df.ConvertedSalary))


# In[91]:


plotly_data = [
    plotly.graph_objs.Histogram(
        x=df.ConvertedSalary.fillna(np.nanmean(df.ConvertedSalary))
    ),
]
layout = plotly.graph_objs.Layout(width=700, height=400)
fig = plotly.graph_objs.Figure(data=plotly_data, layout=layout)
plotly.offline.plot(fig)  # 図2.11 欠損値のすべてに同じ値を入れたときの分布

