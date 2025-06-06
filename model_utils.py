from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil

from data_preprocessing import load_data

def evaluate_models(data_train, y_train, data_test, y_test):
    """
    使用三种模型（KNN、Logistic回归、决策树）对传入的数据进行训练与评估。

    参数：
    - data_train: list or array-like, shape (n_samples, n_features)
    - y_train: list or array-like, shape (n_samples, n_outputs)
    - data_test: same as data_train
    - y_test: same as y_train

    返回：
    - results_dict: dict, 每个模型在每个输出维度的准确率
    """
    X_train = np.squeeze(np.array(data_train))
    X_test = np.squeeze(np.array(data_test))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 保证 y 是二维的（无论是一列还是多列）
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    n_outputs = y_train.shape[1]
    results_dict = {}

    models = {
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    for name, model in models.items():
        model_results = []
        print(f"{name} evaluation begins...")

        output_dir = f"output/{name}/{n_outputs}"
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(n_outputs):
            model.fit(X_train, y_train[:, i])
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test[:, i], y_pred)
            model_results.append(acc)

            # print(f"准确率: {acc:.4f}")
            with open(f"{output_dir}/accuracy_{i + 1}.txt", 'w') as f:
                f.write(f"{acc:.4f}")
            cls_report = classification_report(y_test[:, i], y_pred, zero_division=0, output_dict=True)
            cls_report = pd.DataFrame(cls_report).transpose()
            cls_report.to_csv(f"{output_dir}/cls_report_{i + 1}.csv")

            labels = np.unique(np.concatenate([y_test[:, i], y_pred]))
            cm = confusion_matrix(y_test[:, i], y_pred, labels=labels)

            # cm = confusion_matrix(y_test[:, i], y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title(f"{name} Confusion Matrix (output {i+1})")
            plt.xlabel("Predicted")
            plt.ylabel("Truth Ground")
            plt.tight_layout()
            # plt.show()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/confusion_matrix_{i + 1}.png')

        results_dict[name] = model_results if len(model_results) > 1 else model_results[0]

    return results_dict


if __name__ == "__main__":
    # DataLoader
    data_train, y_train, data_test, y_test = load_data(predict_kws=['G1', 'G2', 'G3'])
    # Evaluation
    results = evaluate_models(data_train, y_train, data_test, y_test)
