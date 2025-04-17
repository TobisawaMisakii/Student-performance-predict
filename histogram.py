'''
This script divides students into 5 groups based on their scores, and give out histogram
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def histogram(data, classification_kws):
    bar_width = 1
    bin_indices = np.arange(len(classification_kws) - 1)  # indices

    # Calculate histogram manually
    counts, _ = np.histogram(data, bins=classification_kws)

    plt.bar(bin_indices, counts, width=bar_width, edgecolor='black', align='center')
    plt.xticks(bin_indices, [f"{classification_kws[i]}-{classification_kws[i+1]}" for i in range(len(classification_kws) - 1)])
    plt.title('Histogram of Student Grades')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    # plt.show()
    plt.savefig('output/histogram.png')

if __name__ == "__main__":
    data = pd.read_csv('data/student-por.csv')
    data = np.array(data)

    G3_grades = [int(data[person][0].split(';')[-1]) for person in range(len(data))]


    sorted_grades = sorted(G3_grades)
    classification_kws = [sorted_grades[i] for i in range(0, len(sorted_grades), len(sorted_grades)//5)]

    histogram(G3_grades, classification_kws)
