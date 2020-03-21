'''
Created on 3/21/20

@author: dulanj
'''
from random import randint
import pandas as pd
import numpy as np
from sklearn import preprocessing


class LVQ():
    def __init__(self):
        pass

    def eucledian_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return np.sqrt(distance)

    def get_best_matching_unit(self, codebooks, test_row):
        codebook_dist = list()
        for codebook in codebooks:
            dist = self.eucledian_distance(codebook, test_row)
            codebook_dist.append((codebook, dist))
        codebook_dist.sort(key=lambda tup: tup[1])
        return codebooks.index(codebook_dist[0][0])

    def random_codebook(self, train):
        no_of_features = len(train[0])
        no_of_elements = len(train)
        codebook = [train[randint(0, no_of_elements-1)][i] for i in range(no_of_features)]
        return codebook

    def train_codebooks(self, train, n_codebooks, lrate, epoches):
        codebooks = [self.random_codebook(train) for _ in range(n_codebooks)]
        for epoch in range(epoches):
            sum_error = 0.0
            current_lr = lrate * (1 - epoch/float(epoches))
            for document in train:
                place = self.get_best_matching_unit(codebooks, document)
                bmu = codebooks[place]
                for i in range(len(bmu)-1):
                    error = document[i] - bmu[i]

                    if bmu[-1] == document[-1]:
                        bmu[i] += current_lr * error
                        sum_error += error ** 2
                    else:
                        bmu[i] -= current_lr * error
                codebooks[place] = bmu

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, current_lr, sum_error))

        return codebooks

    def fit(self, df, n_codebooks, lrate, epoches, normalize=True):
        col_names = df.columns

        if normalize:
            min_max_scaler = preprocessing.MinMaxScaler()
            np_scaled = min_max_scaler.fit_transform(df.iloc[:, :-1])
            df_scale = pd.DataFrame(np_scaled, columns=col_names[:-1])
            class_df = df.iloc[:, -1]
            df_scale.reset_index(drop=True, inplace=True)
            class_df.reset_index(drop=True, inplace=True)
            df = pd.concat([df_scale, class_df], axis=1)

        list_of_lists = df.values.tolist()
        codebooks = self.train_codebooks(list_of_lists, n_codebooks, lrate, epoches)
        new_df = pd.DataFrame(codebooks, columns=col_names)
        return new_df


if __name__ == "__main__":
    obj = LVQ()
    # # Test best matching unit function
    # dataset = [[2.7810836, 2.550537003, 0],
    #            [1.465489372, 2.362125076, 0],
    #            [3.396561688, 4.400293529, 0],
    #            [1.38807019, 1.850220317, 0],
    #            [3.06407232, 3.005305973, 0],
    #            [7.627531214, 2.759262235, 1],
    #            [5.332441248, 2.088626775, 1],
    #            [6.922596716, 1.77106367, 1],
    #            [8.675418651, -0.242068655, 1],
    #            [7.673756466, 3.508563011, 1]]
    # row = dataset[5]
    # for ele in dataset:
    #     print(obj.eucledian_distance(ele, row))
    #
    # print(obj.get_best_matching_unit(dataset, row))
    # # print("======")
    # # print(obj.random_codebook(dataset))
    # #
    # obj.train_codebooks(dataset, 2, 0.3, 20)

    colnames = ['Sample_code_number', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
                'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
                'Normal_Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv("breast-cancer-wisconsin.data", names=colnames)

    # Data pre processing
    data = data.replace({'Class': {2: "Benign", 4: "Malignant"}})
    # Replacing the missing values with 1
    # data = data.replace({'?': 1})
    # Remove data wich has missing values
    data = data[data.Bare_Nuclei != "?"]

    total_samples = data['Sample_code_number'].count()
    print("Number of rows\t: {}".format(total_samples))
    cat_vars = ['Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
                'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
                'Normal_Nucleoli', 'Mitoses', 'Class']
    data_final = data[cat_vars]

    # data_final["Bare_Nuclei"] = data_final.to_numeric(df["Bare_Nuclei"])
    data_final = data_final.astype({"Bare_Nuclei": int})
    obj.fit(data_final, 250, 0.3, 20, normalize=True)


