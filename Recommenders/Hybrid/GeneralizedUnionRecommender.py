import numpy as np


class GeneralizedUnionRecommender:

    def __init__(self, URM_train, recommenders):
        self.URM_train = URM_train
        self.recommenders = recommenders

    def recommend(self, user_ids, cutoff=10, remove_seen_flag=True, remove_top_pop_flag=False, return_scores=True,
                  remove_custom_items_flag=True):

        final_recommendations = []
        final_scores = []

        for user_id in user_ids:

            unique_candidates = set()

            for recommender in self.recommenders:
                recs, _ = recommender.recommend(
                    [user_id],
                    cutoff=cutoff,
                    remove_seen_flag=remove_seen_flag,
                    remove_top_pop_flag=remove_top_pop_flag,
                    return_scores=True,
                    remove_custom_items_flag=remove_custom_items_flag
                )

                unique_candidates.update(recs[0])

            current_recs = np.array(list(unique_candidates))
            final_recommendations.append(current_recs)

            if return_scores:
                final_scores.append(np.ones(len(current_recs)))

        if return_scores:
            return np.array(final_recommendations, dtype=object), np.array(final_scores, dtype=object)

        return np.array(final_recommendations, dtype=object)

    def get_URM_train(self):
        return self.URM_train