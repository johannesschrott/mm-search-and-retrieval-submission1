import statistics

import torch

import constants
import similarity
from itertools import combinations
import pandas as pd
from torchmetrics.functional import kendall_rank_corrcoef, spearman_corrcoef
from tqdm import tqdm
import multiprocessing

nr_of_songs = 10

sim_x_feature = []

ids = pd.read_csv(constants.IDS_PATH, sep="\t", index_col=3, header=0)["id"].values

for sim in similarity.SimilarityFunctionType:
    for feature in similarity.FeaturesType:
        sim_x_feature.append((sim, feature))

correlation_matrix = pd.DataFrame(columns=sim_x_feature)
correlation_matrix.insert(0, "id", sim_x_feature, False)
correlation_matrix.set_index("id", inplace=True)

for x in sim_x_feature:
    correlation_matrix.at[x, x] = 1

sim_x_feature_combinations = list(combinations(sim_x_feature,2))

if torch.cuda.is_available():
    # logging.info("Using CUDA")
    print("-------using CUDA-------")
    device = torch.device("cuda:0")
elif torch.has_mps:
    print("-------using MPS--------")
    # logging.info("Using MPS")
    device = torch.device("cpu")
else:
    print("-------using CPU--------")
    device = torch.device("cpu")

print(sim_x_feature_combinations)


def correlation(combination, ids, nr_of_songs, device, combination_correlation_list):
    spearman_list = []
    kendall_list = []
    for i in tqdm(range(0, nr_of_songs)):
        res1 = similarity.doQueryWithId(ids[i], 100, combination[0][1], combination[0][0]).values
        res2 = similarity.doQueryWithId(ids[i], 100, combination[1][1], combination[1][0]).values
        res1 = [(row_id_similarity[0], row_id_similarity[1]) for row_id_similarity in
                res1]
        res2 = [(row_id_similarity[0], row_id_similarity[1]) for row_id_similarity in
                res2]
        retrieved_songs1 = [row_id_similarity[0] for row_id_similarity in res1]
        retrieved_songs2 = [row_id_similarity[0] for row_id_similarity in res2]
        retrieved_sim_vals1 = [
            res1[retrieved_songs1.index(ids[i])][1] if ids[i] in retrieved_songs1 else 0.0 for i in
            range(0, constants.NR_OF_SONGS)]
        retrieved_sim_vals2 = [
            res1[retrieved_songs2.index(ids[i])][1] if ids[i] in retrieved_songs2 else 0.0 for i in
            range(0, constants.NR_OF_SONGS)]
        r1_tensor = torch.tensor(retrieved_sim_vals1)
        r2_tensor = torch.tensor(retrieved_sim_vals2)
        r1_tensor = r1_tensor.to(device)
        r2_tensor = r2_tensor.to(device)
        # res_kendall = kendall_rank_corrcoef(r1_tensor, r2_tensor)
        res_spearman = spearman_corrcoef(r1_tensor, r2_tensor)
        #     res_kendall = res_kendall.to("cpu")
        #res_spearman = res_spearman.to("cpu")
        #   kendall_list.append(res_kendall.item())
        spearman_list.append(res_spearman.item())

    # kendal_total = statistics.mean(kendall_list)
    spearman_total = statistics.mean(spearman_list)
    print("=====================================")
    print(f"Sim_func 1: {combination[0][0]}, feature 1: {combination[0][1]}")
    print(f"Sim_func 2: {combination[1][0]}, feature 2: {combination[1][1]}")
    #  print(f"Kendall (average over {nr_of_songs} songs):  {kendal_total}")
    print(f"Spearman (average over {nr_of_songs} songs): {spearman_total}")
    combination_correlation_list.append((combination, spearman_total))


if __name__ == "__main__":

    #for i in tqdm(range(0, nr_of_songs)):
    with multiprocessing.Manager() as manager:
        combination_correlation_list = manager.list()
        with manager.Pool() as pool:
            print("computing...")
            multipleResults = [pool.apply_async(correlation, args=(combination, ids, nr_of_songs, device,
                                                                   combination_correlation_list))
                               for combination in tqdm(sim_x_feature_combinations)]

            print("Catching results of computation...")
            [result.wait() for result in tqdm(multipleResults)]

        for comb in combination_correlation_list:
            correlation_matrix.at[comb[0][0], comb[0][1]] = comb[1]
            correlation_matrix.at[comb[0][1], comb[0][0]] = comb[1]
        correlation_matrix.to_csv("data/correlation_matrix.csv")

    print("ready")

