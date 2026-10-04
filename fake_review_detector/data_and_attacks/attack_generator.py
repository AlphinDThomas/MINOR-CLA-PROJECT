
import pandas as pd
import numpy as np


def generate_attack(df, target_item, num_fake_users=50,
                    num_other_items=20, seed=42):

    rng = np.random.default_rng(seed)

    target_data = df[df["item_id"] == target_item]

    if len(target_data) == 0:
        raise ValueError("Target item not found.")

    target_average = target_data["rating"].mean()

    other_items = df.loc[
        df["item_id"] != target_item,
        "item_id"
    ].unique()

    fake_rows = []

    for i in range(num_fake_users):

        fake_user_id = f"FAKE_USER_{i+1:03d}"

        selected_items = rng.choice(
            other_items,
            size=num_other_items,
            replace=False
        )

        other_ratings = rng.choice(
            [3.0, 4.0],
            size=num_other_items,
            p=[0.71, 0.29]
        )

        fake_rows.append({
            "user_id": fake_user_id,
            "item_id": target_item,
            "rating": 5.0,
            "timestamp": int(rng.choice(df["timestamp"].values)),
            "is_fake": True
        })

        for item, rating in zip(selected_items, other_ratings):

            fake_rows.append({
                "user_id": fake_user_id,
                "item_id": item,
                "rating": rating,
                "timestamp": int(rng.choice(df["timestamp"].values)),
                "is_fake": True
            })

    fake_df = pd.DataFrame(fake_rows)

    original_df = df.copy()
    original_df["is_fake"] = False

    start_id = original_df["review_id"].max() + 1

    fake_df.insert(
        0,
        "review_id",
        range(start_id, start_id + len(fake_df))
    )

    columns = [
        "review_id",
        "user_id",
        "item_id",
        "rating",
        "timestamp",
        "is_fake"
    ]

    original_df = original_df[columns]
    fake_df = fake_df[columns]

    attacked_df = pd.concat(
        [original_df, fake_df],
        ignore_index=True
    )

    return attacked_df
