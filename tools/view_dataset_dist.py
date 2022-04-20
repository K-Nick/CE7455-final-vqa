import sys

sys.path.insert(0, "..")
from util.io import load_pickle
from torchtext.vocab import Vocab
from tqdm import tqdm
import pandas as pd


def main():
    cache_dict = load_pickle("../cache_bin/data_cache.freq12.pkl", silent=False)
    train_set = cache_dict["train_dataset"]
    val_set = cache_dict["val_dataset"]
    a_vocab: Vocab = cache_dict["a_vocab"]
    q_vocab: Vocab = cache_dict["q_vocab"]

    def extract_qa(dataset):
        qa_list = []
        for img_path, q_idx, a_idx, meta_dict in tqdm(dataset, miniters=100):
            q_str = q_vocab.lookup_tokens(q_idx)
            a_str = a_vocab.lookup_tokens(a_idx)
            for a_str_ in a_str:
                qa_list += [{"q": " ".join(q_str), "a": a_str_} | meta_dict]

        return qa_list

    train_qa_list = extract_qa(train_set)
    val_qa_list = extract_qa(val_set)

    train_df = pd.DataFrame(
        train_qa_list,
    )
    val_df = pd.DataFrame(val_qa_list)

    # https://stackoverflow.com/questions/20461165/how-to-convert-index-of-a-pandas-dataframe-into-a-column
    train_agg_df = train_df.groupby(["q", "a"]).size().to_frame("train_cnt")
    val_agg_df = val_df.groupby(["q", "a"]).size().to_frame("val_cnt")

    join_df = val_agg_df.join(train_agg_df, how="left")

    join_df = join_df.join(
        val_agg_df.reset_index()
        .groupby("q")
        .agg({"val_cnt": "sum"})
        .rename(columns={"val_cnt": "val_q_cnt"})
    )
    join_df = join_df.join(
        train_agg_df.reset_index()
        .groupby("q")
        .agg({"train_cnt": "sum"})
        .rename(columns={"train_cnt": "train_q_cnt"})
    )

    join_df["train_cnt"] = join_df["train_cnt"].fillna(0).astype("int32")
    join_df["train_q_cnt"] = join_df["train_q_cnt"].fillna(1.0e-13)

    join_df["train_ratio"] = join_df["train_cnt"] / join_df["train_q_cnt"]
    join_df["val_ratio"] = join_df["val_cnt"] / join_df["val_q_cnt"]

    join_df = join_df.reset_index()
    join_df.sort_values(
        ["val_q_cnt", "q", "val_ratio"], ascending=[False, True, False], inplace=True
    )

    join_df = join_df.set_index(["q", "a"])

    join_df.to_excel("../cache_bin/train_val_dist_comparision.xlsx")


if __name__ == "__main__":
    main()
