import Collaborative_Filtering_recsys
import Content_based_recsys

Final_rec_book_list = list(set(Content_based_recsys.rec_book_list_by_contentFiltering).intersection(
    Collaborative_Filtering_recsys.rec_book_list_by_collaborativeFiltering))

print(Final_rec_book_list)
print(Content_based_recsys.rec_book_list_by_contentFiltering)
print(Collaborative_Filtering_recsys.rec_book_list_by_collaborativeFiltering)

RL_data_rating_over_4 = Collaborative_Filtering_recsys.RL_data[
    Collaborative_Filtering_recsys.RL_data['rating'] >= 4].drop('Unnamed: 0', axis=1)
index_over_4 = list(RL_data_rating_over_4['item'])

precision_rec_result = list(set(index_over_4).intersection(Final_rec_book_list))
precision_percent = len(precision_rec_result) / len(Final_rec_book_list)
print(precision_percent)

"""
추천 제시 기준은 책의 평점 평균이 높을 수록 먼저 제시해주기
step1. RL_data에 item별 rating을 평균을 내서 dict 형태에 저장하기
step2. 최종 추천 결과에 rating값을 바탕으로 sort 진행
step3. 추천은 top10개만 진행하기
"""

"""
머신러닝 추천 시스템 성능 평가 지표(강화학습과 동일)
1. NDCG
2. DCG
3. rating이 4, 5 이상인 경우 몇 퍼센트인지?
"""