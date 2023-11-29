import gzip
import shutil
import json
import pandas as pd

# unzip the meta_data_json.gz file
with gzip.open('data/meta_Books.json.gz', 'rb') as f_in:
    with open('meta_Books.json', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# meta data parsing and make dataframe and save to csv file
json_list = []
i = 0
for line in open('meta_Books.json'):
    if i <= 831359:
        json_list.append(json.loads(line))
    i += 1

df_md = pd.DataFrame(json_list)
df_md.to_csv("meta_data_book.csv")

# meta data 중 category관련 전처리 진행
df_md = pd.read_csv("meta_data_book.csv")
df_md.drop('Unnamed: 0', axis=1, inplace=True)

df_md_Ca = df_md[df_md['category'] != '[]']
df_md_Ca['category'] = df_md_Ca['category'].str.replace(' &amp; ', ' & ', regex=True)

df_md_CaItTi = df_md_Ca.loc[:, ['category', 'title', 'asin']]
df_md_CaItTi.rename(columns={'asin': 'item'}, inplace=True)
# Item: str to int
df_md_CaItTi.item = pd.Categorical(df_md_CaItTi.item)
df_md_CaItTi['item'] = df_md_CaItTi.item.cat.codes

# data import
df_orig = pd.read_csv('./data/preprocess_data', names=['item', 'user', 'rating', 'timestamp'])

# data merging
df_merged = pd.merge(df_orig, df_md_CaItTi, how='outer', on='item')
df_merged = df_merged.dropna(how='any')
df_merged.to_csv("df_merged.csv")