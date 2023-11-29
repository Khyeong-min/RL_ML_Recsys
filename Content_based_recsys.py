import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import Collaborative_Filtering_recsys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


"""
Content based filtering
"사용자 본인이 읽은 책과 유사한 책을 추천해주는 것"
1) 책 간의 유사도를 측정
-> 유사도 측정 방법: 장르의 BOW를 바탕으로 유사도 측정
2) 사용자 본인이 읽은 책을 바탕으로 가장 유사한 책들을 추천
=> 결국, 본인이 읽은 책 중에 장르가 가장 유사한 다른 책을 추천해준다.

input: 사용자 본인이 읽은 책에 대한 정보
output: 사용자 본인이 읽은 책과 장르상으로 유사한 책 추천
"""

RL_data_for_Content_based = Collaborative_Filtering_recsys.RL_data

# 책의 장르만을 통해서 어떻게 장르가 분포되어 있는지를 바탕으로 데이터프레임 형성
RL_data_for_Content_based = RL_data_for_Content_based[['item', 'category']]
RL_data_for_Content_based = RL_data_for_Content_based.drop_duplicates()

# item과 category를 통해 DF 생성
item_category_df = pd.DataFrame(index=RL_data_for_Content_based['item'],
                                columns=Collaborative_Filtering_recsys.real_category)

for i in range(len(RL_data_for_Content_based)):
    i = RL_data_for_Content_based.index[i]
    # 기존의 RL_Data에서 category만 뽑기
    for each_category in RL_data_for_Content_based['category'][i][1:-1].split(','):
        item_category_df.loc[RL_data_for_Content_based['item'][i], each_category] = 1

item_category_df = item_category_df.fillna(0)
item_category_df.to_csv("책과 카테고리에 대한 최종 DF.csv")

# 코사인 유사도 구하기
# 인덱스 번호(0부터 7까지 순서대로)
item_category_cosine_DF = pd.DataFrame(cosine_similarity(item_category_df, item_category_df),
                                       index=item_category_df.index, columns=item_category_df.index)
print(item_category_cosine_DF)

# 유저가 읽은 책을 바탕으로 가장 유사한 책을 추천해주는 것
input_user_idx = int(input("user_idx?"))
print("유저가 읽은 책과 가장 유사한 책들을 추천한 결과는? ")
# 해당 유저가 읽은 책의 index를 리스트에 저장
user_book_list = []
for i in range(28826):
    column_idx = Collaborative_Filtering_recsys.user_book_df.columns[i]
    if Collaborative_Filtering_recsys.user_book_df.loc[
        Collaborative_Filtering_recsys.idx_dict[input_user_idx], column_idx] == 1:
        user_book_list.append(column_idx)

# 해당 유저가 읽은 책과 가장 유사한 책들을 출력
rec_book_list_by_contentFiltering = []
for book_index in user_book_list:
    similar_book_index = list(item_category_cosine_DF.loc[book_index].nlargest(2, keep='first').index)[-1]
    if (book_index != similar_book_index) and (similar_book_index not in rec_book_list_by_contentFiltering):
        rec_book_list_by_contentFiltering.append(similar_book_index)

print(rec_book_list_by_contentFiltering)