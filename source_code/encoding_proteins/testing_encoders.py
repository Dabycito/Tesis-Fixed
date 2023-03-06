import sys
import pandas as pd

print("Preparing imports")

from physicochemical_properties import physicochemical_encoder
from fft_encoding import fft_encoding
from constant_values import constant_values

# console params
print("Get params by console")
dataset = pd.read_csv(sys.argv[1])
property_to_use = sys.argv[2]
encoders = pd.read_csv(sys.argv[3])
encoders.index = encoders['residue']

path_export = sys.argv[4]
column_with_seq = sys.argv[5]
column_with_id = sys.argv[6]

print("Instance object")
encoding_instance = physicochemical_encoder(dataset,
                 property_to_use,
                 encoders,
                 constant_values(),
                 column_with_seq,
                 column_with_id)

print("Start encoding process")
name_export = "{}{}_encoding.csv".format(path_export, property_to_use)
is_export = True

df_data = encoding_instance.encoding_dataset(
            name_export,
            is_export)


print("Encoding FFT")
fft_encoding_instance = fft_encoding(df_data,
                                     len(df_data.columns)-1,
                                     column_with_id,
                                     constant_values())

name_export = "{}{}_encoding_FFT.csv".format(path_export, property_to_use)
df_data_2 = fft_encoding_instance.encoding_dataset(
    name_export,
    is_export)

print("End script")

