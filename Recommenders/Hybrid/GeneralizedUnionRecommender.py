import numpy as np
import math


class GeneralizedUnionRecommender:

    def __init__(self, URM_train, recommenders):
        self.URM_train = URM_train
        self.recommenders = recommenders
        self.n_recommenders = len(recommenders)

    def recommend(self, user_ids, cutoff=None, remove_seen_flag=True, remove_top_pop_flag=False, return_scores=True,
                  remove_custom_items_flag=True):

        safe_cutoff = cutoff if cutoff is not None and cutoff > 0 else 50

        quota_per_model = safe_cutoff // self.n_recommenders

        if quota_per_model < 1:
            quota_per_model = 1

        final_recommendations = []

        for user_id in user_ids:
            unique_candidates = set()

            for recommender in self.recommenders:
                recs, _ = recommender.recommend(
                    [user_id],
                    cutoff=quota_per_model,
                    remove_seen_flag=remove_seen_flag,
                    remove_top_pop_flag=remove_top_pop_flag,
                    return_scores=True,
                    remove_custom_items_flag=remove_custom_items_flag
                )
                unique_candidates.update(recs[0])

            final_recommendations.append(np.array(list(unique_candidates)))

        if return_scores:
            _, final_scores_matrix = self.recommenders[0].recommend(
                user_ids,
                cutoff=cutoff,
                remove_seen_flag=remove_seen_flag,
                remove_top_pop_flag=remove_top_pop_flag,
                return_scores=True,
                remove_custom_items_flag=remove_custom_items_flag
            )
            return np.array(final_recommendations, dtype=object), final_scores_matrix

        return np.array(final_recommendations, dtype=object)

    def get_URM_train(self):
        return self.URM_train