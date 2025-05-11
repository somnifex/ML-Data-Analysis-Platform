import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
import openpyxl
import random
import os

N_SAMPLES = 200  # 生成的样本数量
N_FEATURES_NUMERICAL = 5  # 数值特征的数量
N_FEATURES_CATEGORICAL = 2  # 类别特征的数量 (用于混合数据集)
OUTPUT_DIR = "ml_test_data"  # 输出文件夹名称

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def generate_binary_classification_data(
    n_samples=N_SAMPLES, n_features=N_FEATURES_NUMERICAL, random_state=42
):
    """生成二元分类数据"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),  # 信息特征数量
        n_redundant=max(0, n_features // 3),  # 冗余特征数量
        n_classes=2,
        random_state=random_state,
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    print(
        f"Generated binary classification data with {n_samples} samples and {n_features} features."
    )
    return df


def generate_multiclass_classification_data(
    n_samples=N_SAMPLES, n_features=N_FEATURES_NUMERICAL, n_classes=3, random_state=42
):
    """生成多元分类数据"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(0, n_features // 3),
        n_classes=n_classes,
        n_clusters_per_class=1,  # 每个类别有几个簇
        random_state=random_state,
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    print(
        f"Generated multi-class classification data with {n_samples} samples, {n_features} features, and {n_classes} classes."
    )
    return df

def generate_regression_data(
    n_samples=N_SAMPLES, n_features=N_FEATURES_NUMERICAL, noise=10.0, random_state=42
):
    """生成回归数据"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        noise=noise,
        random_state=random_state,
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    print(
        f"Generated regression data with {n_samples} samples and {n_features} features."
    )
    return df


# --- 4. 生成聚类数据 (无标签) ---
def generate_clustering_data(
    n_samples=N_SAMPLES, n_features=2, n_centers=3, random_state=42
):
    """生成聚类数据 (用于无监督学习)"""
    X, _ = make_blobs( 
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=1.0,  # 簇的标准差
        random_state=random_state,
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    print(
        f"Generated clustering data with {n_samples} samples, {n_features} features, and {n_centers} centers."
    )
    return df


def generate_mixed_data(
    n_samples=N_SAMPLES,
    n_numerical=N_FEATURES_NUMERICAL,
    n_categorical=N_FEATURES_CATEGORICAL,
    random_state=42,
):
    """生成包含数值和类别特征的混合数据，用于分类任务"""
    np.random.seed(random_state)
    random.seed(random_state)

    data = {}
    # 生成数值特征
    for i in range(n_numerical):
        if i % 2 == 0:
            data[f"num_feature_{i+1}"] = np.random.randn(n_samples) * random.randint(
                5, 15
            ) + random.randint(
                -10, 10
            )  # 正态分布
        else:
            data[f"num_feature_{i+1}"] = np.random.rand(n_samples) * random.randint(
                20, 100
            )

    # 生成类别特征
    categorical_options = {
        "cat_feature_1": ["A", "B", "C", "D"],
        "cat_feature_2": ["Low", "Medium", "High"],
        "cat_feature_3": ["Type1", "Type2", "Type3", "Type4", "Type5"],
    }
    cat_keys = list(categorical_options.keys())
    for i in range(n_categorical):
        key_to_use = cat_keys[i % len(cat_keys)] 
        data[key_to_use] = [
            random.choice(categorical_options[key_to_use]) for _ in range(n_samples)
        ]

    df = pd.DataFrame(data)

    target = []
    for i in range(n_samples):
        score = 0
        if df["num_feature_1"].iloc[i] > np.median(
            df["num_feature_1"]
        ): 
            score += 1
        if (
            df[cat_keys[0]].iloc[i]
            in categorical_options[cat_keys[0]][
                : len(categorical_options[cat_keys[0]]) // 2
            ]
        ): 
            score += 1

        if score >= 1.5 and random.random() < 0.7:  # 满足条件，70%概率为1
            target.append(1)
        elif score >= 0.5 and random.random() < 0.4:  # 部分满足条件，40%概率为1
            target.append(1)
        elif random.random() < 0.15:  # 不太满足条件，15%概率为1
            target.append(1)
        else:
            target.append(0)

    df["target"] = target
    print(
        f"Generated mixed data with {n_samples} samples, {n_numerical} numerical and {n_categorical} categorical features."
    )
    return df


if __name__ == "__main__":
    df_binary_clf = generate_binary_classification_data(n_features=5)
    df_binary_clf.to_excel(
        os.path.join(OUTPUT_DIR, "binary_classification_data.xlsx"), index=False
    )
    print(f"Saved binary_classification_data.xlsx to {OUTPUT_DIR}\n")

    df_multiclass_clf = generate_multiclass_classification_data(
        n_features=6, n_classes=4
    )
    df_multiclass_clf.to_excel(
        os.path.join(OUTPUT_DIR, "multiclass_classification_data.xlsx"), index=False
    )
    print(f"Saved multiclass_classification_data.xlsx to {OUTPUT_DIR}\n")

    df_regression = generate_regression_data(n_features=4, noise=20)
    df_regression.to_excel(
        os.path.join(OUTPUT_DIR, "regression_data.xlsx"), index=False
    )
    print(f"Saved regression_data.xlsx to {OUTPUT_DIR}\n")

    df_clustering = generate_clustering_data(
        n_features=3, n_centers=4
    )
    df_clustering.to_excel(
        os.path.join(OUTPUT_DIR, "clustering_data.xlsx"), index=False
    )
    print(f"Saved clustering_data.xlsx to {OUTPUT_DIR}\n")

    df_mixed = generate_mixed_data(n_numerical=4, n_categorical=2)
    df_mixed.to_excel(
        os.path.join(OUTPUT_DIR, "mixed_feature_classification_data.xlsx"), index=False
    )
    print(f"Saved mixed_feature_classification_data.xlsx to {OUTPUT_DIR}\n")

    print(f"All test Excel files have been generated in the '{OUTPUT_DIR}' directory.")
