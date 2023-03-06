import pandas as pd
from utils_functions import utils_functions
from joblib import Parallel, delayed

class distance_estimator(object):

    def __init__(self,
                 dataset,
                 type_distance,
                 constant_instance,
                 response_name,
                 id_seq_name):

        self.dataset = dataset
        self.type_distance = type_distance
        self.response_name = response_name
        self.id_seq_name = id_seq_name
        self.constant_instance = constant_instance

    def __get_row(self, index_data):
        row = [self.dataset[column][index_data] for column in self.dataset.columns if column not in [self.response_name, self.id_seq_name]]
        return row

    def __distance_sequence_to_all(self, index_data):

        row_target = self.__get_row(index_data)
        distance_values = []

        for i in range(len(self.dataset)):
            if i != index_data:
                row_evaluate = self.__get_row(i)
                distance_value = utils_functions().get_distance_vectors(row_target, row_evaluate, self.type_distance)
                row_value = [
                    self.dataset[self.id_seq_name][index_data],
                    self.dataset[self.response_name][index_data],
                    self.dataset[self.id_seq_name][i],
                    self.dataset[self.response_name][i],
                    distance_value
                ]
                distance_values.append(row_value)
        return distance_values

    def generate_matrix_distance(self, name_export=None, is_export=None):

        print("Start distance calculator")
        response_distance = Parallel(n_jobs=self.constant_instance.n_cores, require='sharedmem')(delayed(self.__distance_sequence_to_all)(i) for i in range(len(self.dataset)))

        print("Process response")
        matrix_data = []
        for element in response_distance:
            for row in element:
                matrix_data.append(row)

        header = [
            self.id_seq_name+"_seq1",
            self.response_name+"_seq1",
            self.id_seq_name + "_seq2",
            self.response_name + "_seq2",
            "distance"
        ]
        df_export = pd.DataFrame(matrix_data, columns=header)

        if is_export:
            print(name_export)
            utils_functions().export_csv(df_export, name_export)

        return df_export