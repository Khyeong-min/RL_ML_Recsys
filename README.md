## Book Recommendation System using RL(DDPG) and ML(hybrid Filtering)

## About Code
- Collaborative_Filtering_recsys: user-based Filtering(cosine_similarity)
- Content_based_recsys: content based Filtering(cosine_similarity)
- Data_prep: Data preprocessing(rating >=20, user >= 10)
- DDPG_recsys: recommendation by DDPG algorithm
- metatdata_applied: next step after finishing Data_prep which preprocess the added data
- ML_recsys_merged: merge Collaborative Filtering with Content based

## About Dataset
- [Amazon: Book.csv](https://nijianmo.github.io/amazon/index.html#subsets)
  - book_3000_new.csv: file size-> 83385, column-> {user, item, rating, timestamp} 
  - metadata: asin, title, feature, description, price, image, related, salesRank, rand, categories, tech1, tech2, similar

## Performance Evaluation 
- Evaluation Metrics: DCG, NDCG, Precision_rating_over_4