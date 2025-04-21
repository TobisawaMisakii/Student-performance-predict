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

def evaluate_models(data_train, y_train, data_test, y_test, title=""):
    """
    使用三种模型（KNN、Logistic回归、决策树）对传入的数据进行训练与评估。

    参数：
    - data_train: list or array-like, shape (n_samples, n_features)
    - y_train: list or array-like, shape (n_samples, n_outputs)
    - data_test: same as data_train
    - y_test: same as y_train
    - title: string, 标题显示用

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
        print(f"======================")
        print(f"{name}:")

        for i in range(n_outputs):
            print(f"第 {i+1} 个输出")
            model.fit(X_train, y_train[:, i])
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test[:, i], y_pred)
            model_results.append(acc)

            print(f"准确率: {acc:.4f}")
            cls_report = classification_report(y_test[:, i], y_pred, zero_division=0, output_dict=True)
            cls_report = pd.DataFrame(cls_report).transpose()
            shutil.rmtree(f"output/{name}", ignore_errors=True)
            os.makedirs(f"output/{name}", exist_ok=True)
            cls_report.to_csv(f"output/{name}/cls_report_{i + 1}.csv")

            labels = np.unique(np.concatenate([y_test[:, i], y_pred]))
            cm = confusion_matrix(y_test[:, i], y_pred, labels=labels)

            # cm = confusion_matrix(y_test[:, i], y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title(f"{name} Confusion Matrix (output {i+1})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'output/{name}/confusion_matrix_{i + 1}.png')

        results_dict[name] = model_results if len(model_results) > 1 else model_results[0]

    return results_dict


if __name__ == "__main__":
    # DataLoader
    data_train, y_train, data_test, y_test = load_data(predict_kws=['G3'])
    # Evaluation
    results = evaluate_models(data_train, y_train, data_test, y_test, title="Student Performance")

    print("======================")

