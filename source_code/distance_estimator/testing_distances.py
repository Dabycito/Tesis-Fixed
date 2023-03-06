import pandas as pd
import sys
print("Preparing imports")

from distance_estimator import distance_estimator
from constant_values import constant_values

print("Console params")
input_data = pd.read_csv(sys.argv[1])
name_export = sys.argv[2]
column_with_response = sys.argv[3]
column_with_id = sys.argv[4]

distance_instance = distance_estimator(input_data,
                                       1,
                                       constant_values(),
                                       column_with_response,
                                       column_with_id)

distance_instance.generate_matrix_distance(name_export=name_export,
                                           is_export=True)