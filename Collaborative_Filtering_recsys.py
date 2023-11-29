import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 데이터 불러오기
RL_data = pd.read_csv("df_merged.csv")

"""
유저 기반 collaborative filtering
"유사한 사용자의 데이터를 기반으로 추천해주기"
1)사용자가 읽은 책의 장르를 BOW를 통해 리스트로 만들고 이를 바탕으로 사용자 간에 유사도 측정
2) 사용자가 독서한 책이 얼마나 겹치는지 확률로 측정(두 사람이 읽은 책 중에 겹치는 책의 갯수/두 사람이 읽은 모든 책의 갯수)=> 0부터 1사이에 대하여 유사도를 뽑기
-> 동시 적용: 1번과 2번의 확률에 대한 값을 통해서 두 사람간의 유사도를 측정
=> 결국, 최종 유사도를 측정한 것을 바탕으로 유사한 사용자는 읽었지만 사용자 본인은 읽지 않은 책을 추천해준다.
"""

"""
사용자가 읽은 책의 장르를 BOW를 통해 리스트로 만들고 이를 바탕으로 사용자 간에 유사도 측정(RL_data_for_Collaborative_Filtering1)
input: 사용자가 읽은 책의 장르에 대한 정보
output: 사용자가 읽은 책의 장르를 바탕으로 사용자간의 유사도 측정(cosine_similarity)
"""

RL_data_for_Collaborative_Filtering1 = RL_data.drop("Unnamed: 0", axis=1)
# 전체 카테고리 출력
real_category = []

for i in range(len(RL_data)):
    category_list = RL_data_for_Collaborative_Filtering1['category'][i][1:-1].split(',')
    for category in category_list:
        if category not in real_category:
            real_category.append(category)

# 유저를 row, 장르를 column으로 데이터프레임 생성
user_category_df = pd.DataFrame(index=RL_data_for_Collaborative_Filtering1['user'], columns=real_category)
user_category_df = user_category_df.fillna(0)

for i in range(len(RL_data_for_Collaborative_Filtering1)):
    # 기존의 RL_Data에서 category만 뽑기
    for each_category in RL_data_for_Collaborative_Filtering1['category'][i][1:-1].split(','):
        user_category_df.loc[RL_data_for_Collaborative_Filtering1['user'][i], each_category] = 1

# 유저와 장르에 대한 최종 CSV-> 유저가 본 책의 장르이면: +/ 유저가 한 번도 본 책의 장르가 아니면: 0
user_category_df = user_category_df.loc[~user_category_df.index.duplicated(keep='first')]
user_category_df.to_csv('유저와 장르에 대한 최종 DF.csv')

# 코사인 유사도 구하기
# 인덱스 번호(0부터 7까지 순서대로): 16.0, 4.0, 18.0, 13.0, 20.0, 6.0, 12.0, 0.0
user_category_cosine_DF = pd.DataFrame(cosine_similarity(user_category_df, user_category_df))

"""
각 사용자가 읽은 책 목록을 담을 리스트를 만들고 이를 바탕으로 사용자 간의 유사도 측정(RL_data_for_Collaborative_Filtering2)
input: 사용자가 읽은 책의 목록에 대한 정보
output: 사용자가 읽은 책의 목록을 바탕으로 사용자 간의 유사도 측정(cosine_similarity)
"""

# 사용자가 독서한 책 모두를 list 형태로 뽑기(sort 진행)
# 현재 존재하는 사용자의 번호: 16.0, 4.0, 18.0, 13.0, 20.0, 6.0, 12.0, 0.0
RL_data_for_Collaborative_Filtering2 = RL_data[['item', 'user']]

user_16_data = []
user_4_data = []
user_18_data = []
user_13_data = []
user_20_data = []
user_6_data = []
user_12_data = []
user_0_data = []

for idx in range(len(RL_data_for_Collaborative_Filtering2)):
    if RL_data_for_Collaborative_Filtering2['user'][idx] == 16.0:
        user_16_data.append(RL_data_for_Collaborative_Filtering2['item'][idx])
    if RL_data_for_Collaborative_Filtering2['user'][idx] == 4.0:
        user_4_data.append(RL_data_for_Collaborative_Filtering2['item'][idx])
    if RL_data_for_Collaborative_Filtering2['user'][idx] == 18.0:
        user_18_data.append(RL_data_for_Collaborative_Filtering2['item'][idx])
    if RL_data_for_Collaborative_Filtering2['user'][idx] == 13.0:
        user_13_data.append(RL_data_for_Collaborative_Filtering2['item'][idx])
    if RL_data_for_Collaborative_Filtering2['user'][idx] == 20.0:
        user_20_data.append(RL_data_for_Collaborative_Filtering2['item'][idx])
    if RL_data_for_Collaborative_Filtering2['user'][idx] == 6.0:
        user_6_data.append(RL_data_for_Collaborative_Filtering2['item'][idx])
    if RL_data_for_Collaborative_Filtering2['user'][idx] == 12.0:
        user_12_data.append(RL_data_for_Collaborative_Filtering2['item'][idx])
    if RL_data_for_Collaborative_Filtering2['user'][idx] == 0.0:
        user_0_data.append(RL_data_for_Collaborative_Filtering2['item'][idx])

# 모든 유저가 읽은 책을 중복제거하고 합친 순수한 결과
total_read_book_idx_list = []
read_book_idx_list = user_16_data + user_4_data + user_18_data + user_13_data + user_20_data + user_6_data + user_12_data + user_0_data
read_book_idx_list = set(read_book_idx_list)

for book_idx in read_book_idx_list:
    total_read_book_idx_list.append(book_idx)

# 유저가 읽은 책을 바탕으로 DF만들고 cosine_similarity 계산
user_book_df = pd.DataFrame(index=[16.0, 4.0, 18.0, 13.0, 20.0, 6.0, 12.0, 0.0], columns=total_read_book_idx_list)
user_book_df = user_book_df.fillna(0)

for i in range(len(user_book_df)):
    if i == 0:
        for book_idx in user_16_data:
            user_book_df.loc[16.0, book_idx] = 1
    if i == 1:
        for book_idx in user_4_data:
            user_book_df.loc[4.0, book_idx] = 1
    if i == 2:
        for book_idx in user_18_data:
            user_book_df.loc[18.0, book_idx] = 1
    if i == 3:
        for book_idx in user_13_data:
            user_book_df.loc[13.0, book_idx] = 1
    if i == 4:
        for book_idx in user_20_data:
            user_book_df.loc[20.0, book_idx] = 1
    if i == 5:
        for book_idx in user_6_data:
            user_book_df.loc[6.0, book_idx] = 1
    if i == 6:
        for book_idx in user_12_data:
            user_book_df.loc[12.0, book_idx] = 1
    if i == 7:
        for book_idx in user_0_data:
            user_book_df.loc[0.0, book_idx] = 1

# 유저와 책에 대한 최종 CSV-> 유저가 본 책이면: 1/ 유저가 본 책이 아니면: 0
user_book_df.to_csv('유저와 책에 대한 최종 DF.csv')

# 코사인 유사도 구하기
# 인덱스 번호(0부터 7까지 순서대로): 16.0, 4.0, 18.0, 13.0, 20.0, 6.0, 12.0, 0.0
user_book_cosine_DF = pd.DataFrame(cosine_similarity(user_book_df, user_book_df))

# user_category_cosine_df와 user_book_cosine_df를 합친 결과-> DF_for_collaborative_filtering
cosineSim_for_collaborative_filtering = pd.DataFrame(index=[0, 1, 2, 3, 4, 5, 6, 7], columns=[0, 1, 2, 3, 4, 5, 6, 7])
cosineSim_for_collaborative_filtering = cosineSim_for_collaborative_filtering.fillna(0)
for i in range(8):
    for j in range(8):
        cosineSim_for_collaborative_filtering.loc[i, j] = (user_book_cosine_DF.loc[i, j] + user_category_cosine_DF.loc[
            i, j]) / 2
print(cosineSim_for_collaborative_filtering)

# 실제 collarborative filtering 진행
# 인덱싱을 편하게 하기 위한 dictation 선언
idx_dict = {
    0: 16.0,
    1: 4.0,
    2: 18.0,
    3: 13.0,
    4: 20.0,
    5: 6.0,
    6: 12.0,
    7: 0.0
}

# 가장 유사한 유저를 CosineSim_for_collaborative_filtering로 찾기
input_idx = int(input("user_idx?"))
print(input_idx, "과 가장 유사한 유저를 통해 추천된 책의 인덱스 번호는?")
similar_user_idx = 0
base_similarity = 0.0
for i in range(len(cosineSim_for_collaborative_filtering)):
    if i != input_idx and cosineSim_for_collaborative_filtering.loc[input_idx, i] >= base_similarity:
        base_similarity = cosineSim_for_collaborative_filtering.loc[input_idx, i]
        similar_user_idx = i

# 유저와 책에 대한 최종 DF"에 접근해서 사용자는 읽지 않았는데 유사한 유저는 읽는 책 제목을 출력하기
# 유저와 유사한 사용자가 읽은 책들 체크
user_read_book_idx = []
similar_user_read_book_idx = []
rec_book_list_by_collaborativeFiltering = []

for i in range(len(list(user_book_df.loc[idx_dict[input_idx]]))):
    if list(user_book_df.loc[idx_dict[input_idx]])[i] == 1:
        user_read_book_idx.append(total_read_book_idx_list[i])
for j in range(len(list(user_book_df.loc[idx_dict[similar_user_idx]]))):
    if list(user_book_df.loc[idx_dict[similar_user_idx]])[j] == 1:
        similar_user_read_book_idx.append(total_read_book_idx_list[j])

# 유저가 읽지 않은 책 중에 유사한 사용자가 읽은 책 추천
rec_book_list_by_collaborativeFiltering = [x for x in user_read_book_idx if x not in similar_user_read_book_idx]
print(rec_book_list_by_collaborativeFiltering)