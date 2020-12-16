
from A1 import taska
from B1 import taskb

#Task A paths
label_path_a = './Datasets/celeba/labels.csv'
test_label_path_a = './Datasets/celeba_test/labels.csv'
data_path_a = './Datasets/celeba/img/'
test_data_path_a = './Datasets/celeba_test/img'
model_path_a1 = './A1/a1.h5'
model_path_a2 = './A2/a2.h5'

#Task B paths
label_path_b = './Datasets/cartoon_set/labels.csv'
test_label_path_b = './Datasets/cartoon_set_test/labels.csv'
data_path_b = './Datasets/cartoon_set/img/'
test_data_path_b = './Datasets/cartoon_set_test/img'
model_path_b1 = './B1/b1.h5'
model_path_b2 = './B2/b2.h5'
# ======================================================================================================================
# Data preprocessing for Task A
label_a = taska.read_label(label_path_a)
X_a = taska.import_train_data(label_a, data_path_a)
y_a1 = taska.y_a1(label_a)
y_a2 = taska.y_a2(label_a)
X_train_a1, X_valid_a1, y_train_a1, y_valid_a1 = taska.train_valid_split(X_a, y_a1)
X_train_a2, X_valid_a2, y_train_a2, y_valid_a2 = taska.train_valid_split(X_a, y_a2)

test_label_a = taska.read_label(test_label_path_a)
test_X_a = taska.import_test_data(test_label_a, test_data_path_a)
test_y_a1 = taska.y_a1(test_label_a)
test_y_a2 = taska.y_a2(test_label_a)

# Data preprocessing for Task B
label_b = taska.read_label(label_path_b)
X_b = taskb.import_train_data_b(label_b, data_path_b)
y_b1 = taskb.y_b1(label_b)
y_b2 = taskb.y_b2(label_b)
X_train_b1, X_valid_b1, y_train_b1, y_valid_b1 = taska.train_valid_split(X_b, y_b1)
X_train_b2, X_valid_b2, y_train_b2, y_valid_b2 = taska.train_valid_split(X_b, y_b2)

test_label_b = taska.read_label(test_label_path_b)
test_X_b = taskb.import_test_data_b(test_label_b, test_data_path_b)
test_y_b1 = taskb.y_b1(test_label_b)
test_y_b2 = taskb.y_b2(test_label_b)

# ======================================================================================================================
# Task A1
model_A1 = taska.build_model()
acc_A1_train = taska.train_save(model_A1, X_train_a1, X_valid_a1, y_train_a1, y_valid_a1, model_path_a1)
acc_A1_test = taska.evaluate(model_path_a1, test_X_a, test_y_a1)


# ======================================================================================================================
# Task A2
model_A2 = taska.build_model()
acc_A2_train = taska.train_save(model_A2, X_train_a2, X_valid_a2, y_train_a2, y_valid_a2, model_path_a2)
acc_A2_test = taska.evaluate(model_path_a2, test_X_a, test_y_a2)


# ======================================================================================================================
# Task B1
model_B1 = taskb.build_model_b()
acc_B1_train = taska.train_save(model_B1, X_train_b1, X_valid_b1, y_train_b1, y_valid_b1, model_path_b1)
acc_B1_test = taska.evaluate(model_path_b1, test_X_b, test_y_b1)


# ======================================================================================================================
# Task B2
model_B2 = taskb.build_model_b()
acc_B2_train = taska.train_save(model_B2, X_train_b2, X_valid_b2, y_train_b2, y_valid_b2, model_path_b2)
acc_B2_test = taska.evaluate(model_path_b2, test_X_b, test_y_b2)


# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))