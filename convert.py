import csv
csv_file = 'list_attr_celeba.csv'
txt_file = 'list_attr_celeba.txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()