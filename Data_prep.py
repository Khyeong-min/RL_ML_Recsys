import pandas as pd

data = pd.read_csv('data/books_3000_new.csv', names=['item', 'user', 'rating', 'timestamp'])

# User: str to int
data.user = pd.Categorical(data.user)
data['user'] = data.user.cat.codes

# Item: str to int
data.item = pd.Categorical(data.item)
data['item'] = data.item.cat.codes


def preprocess(data, name):
    item_counts = pd.DataFrame(data['item'].value_counts())
    item_indices = item_counts[item_counts['count'] >= 10].index
    data_10 = data[data['item'].isin(item_indices)]

    user_counts = pd.DataFrame(data_10['user'].value_counts())
    user_indices = user_counts[user_counts['count'] >= 20].index
    data_20_10 = data[data['user'].isin(user_indices)]

    latest_data = data_20_10.drop_duplicates(['user', 'item'], keep='first')

    latest_data.to_csv('./data/' + name, index=False, header=None)
    return latest_data


# 최종 df-> item이 10개 이상이며 그 중 20명 이상의 user만을 추출
preprocess(data=data, name='preprocess_data')
book_data = pd.read_csv('./data/preprocess_data')
