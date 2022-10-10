from os import listdir
from os.path import isfile, join

data_path = "../../Data/Full_Transition/"
output_path = "../../Data/Datasets/dataset_16x16_Full_Transition_val.txt"

def remove_duplicates(data):
    for index in range(len(data) - 1, -1, -1):
        if data[index] == data[index - 1]:
            del data[index]
        if index % 10000 == 0:
            print(index)
    return data

all_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

string = ""
len(string)
for filename in all_files:
    f = open(data_path + filename, "r")
    string += f.read()
    f.close()


data_points = string.split("\n")

print(len(data_points))

data_points.sort()

print("Sorted")
data_points = remove_duplicates(data_points)
print(len(data_points))

string = ""
for datapoint in data_points:
    # datapoint = datapoint.replace('n', '-1')
    # datapoint = datapoint.replace('e', '0')
    # datapoint = datapoint.replace('w', '1')
    # datapoint = datapoint.replace('k', '2')
    # datapoint = datapoint.replace('d', '3')
    # datapoint = datapoint.replace('g', '4')
    string += datapoint + "\n"

f = open(output_path, "w")
f.write(string)
f.close()

print("Done")
