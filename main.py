import collections
import math

import pandas as pd
import numpy as np
from os import path
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout, Flatten
# from keras.callbacks import EarlyStopping, TensorBoard
# from keras.optimizers import Adam

TrainingPercent = 80  # dividing training data into 80% of the total
ValidationPercent = 10  # validation data is 10% of the total
TestPercent = 10  # testing data is 10 % of the total
IsSynthetic = False
NumberOfClusters = 10
NUMBER_OF_ITERATIONS = 1000


def readFile(fileName):
    readLine = pd.read_csv(fileName)
    return readLine


def getFeatures(readData, pairs):
    concatArray = []
    subArray = []
    for index, row in pairs.iterrows():
        img_id1 = readData.loc[readData['img_id'] == row['img_id_A']]
        img_id1 = img_id1.loc[:, 'f1':]
        img_id2 = readData.loc[readData['img_id'] == row['img_id_B']]
        img_id2 = img_id2.loc[:, 'f1':]
        concatFeatures = np.concatenate((img_id1.values[0], img_id2.values[0]), axis=0)
        subFeatures = np.absolute(img_id1.values[0] - img_id2.values[0])
        concatArray.append(concatFeatures)
        subArray.append(subFeatures)
    return np.array(concatArray), np.array(subArray)


def generate_raw_data(raw_data, is_synthetic):
    variance_matrix = np.var(raw_data, axis=0)
    if is_synthetic is False:
        zero_variance_index = np.where(variance_matrix == 0)[0]
        raw_data = np.delete(raw_data, zero_variance_index, axis=1)
        variance_matrix = np.delete(variance_matrix, zero_variance_index)
    return raw_data, len(variance_matrix)


def GenerateTrainingTarget(rawTraining, TrainingPercent=80):
    TrainingLen = int(math.ceil(len(rawTraining) * (TrainingPercent * 0.01)))
    t = rawTraining[:TrainingLen]
    return t


def GenerateTrainingDataMatrix(rawData, TrainingPercent=80):
    T_len = int(math.ceil(len(rawData) * 0.01 * TrainingPercent))
    d2 = rawData[0:T_len, :]
    return d2


def GenerateValData(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData) * ValPercent * 0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[TrainingCount + 1:V_End]
    return dataMatrix


def GenerateValTargetVector(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData) * ValPercent * 0.01))
    V_End = TrainingCount + valSize
    t = rawData[TrainingCount + 1:V_End]
    return t


def generate_raw_data_with_relevant_features(raw_data, is_synthetic):
    variance_matrix = np.var(raw_data, axis=0)
    if is_synthetic is False:
        zero_variance_index = np.where(variance_matrix == 0)[0]
        raw_data = np.delete(raw_data, zero_variance_index, axis=1)
        variance_matrix = np.delete(variance_matrix, zero_variance_index)
    return raw_data, len(variance_matrix)


humanObserved = readFile(path.relpath("HumanObserved-Features-Data/HumanObserved-Features-Data.csv"))
human_SamePairs = readFile(path.relpath("HumanObserved-Features-Data/same_pairs.csv"))
human_DiffPairs = readFile(path.relpath("HumanObserved-Features-Data/diffn_pairs.csv"))
human_DiffPairs = human_DiffPairs.sample(frac=1).reset_index(drop=True)
human_DiffPairs = human_DiffPairs.head(791)

# Human Observed Same Pairs.
humanObserved_Same_Features = getFeatures(humanObserved, human_SamePairs)
humanObserved_Features_Concat = humanObserved_Same_Features[0]
humanObserved_Same_Features_Sub = humanObserved_Same_Features[1]
humanObserved_Target_Concat = np.ones(shape=(len(humanObserved_Features_Concat), 1))

# Human Observed different Pairs.
humanObserved_Diff_Features = getFeatures(humanObserved, human_DiffPairs)
humanObserved_Features_Sub = humanObserved_Diff_Features[0]
humanObserved_Diff_Features_Sub = humanObserved_Diff_Features[1]
humanObserved_Target_Sub = np.zeros(shape=(len(humanObserved_Features_Sub), 1))

humanObserved_Features_Concat = np.append(humanObserved_Features_Concat, humanObserved_Target_Concat, axis=1)
humanObserved_Same_Features_Sub = np.append(humanObserved_Same_Features_Sub, humanObserved_Target_Concat, axis=1)
humanObserved_Features_Sub = np.append(humanObserved_Features_Sub, humanObserved_Target_Sub, axis=1)
humanObserved_Diff_Features_Sub = np.append(humanObserved_Diff_Features_Sub, humanObserved_Target_Sub, axis=1)

humanObserved_Features_Concat = np.append(humanObserved_Features_Concat, humanObserved_Features_Sub, axis=0)
humanObserved_Features_Sub = np.append(humanObserved_Same_Features_Sub, humanObserved_Diff_Features_Sub, axis=0)

np.random.shuffle(humanObserved_Features_Concat)
np.random.shuffle(humanObserved_Features_Sub)

humanObserved_Target_Concat = humanObserved_Features_Concat[:, [18]]
humanObserved_Features_Concat = np.delete(humanObserved_Features_Concat, [18], axis=1)
print("humanObserved_Target_Concat", humanObserved_Target_Concat.shape)
print("humanObserved_Features_Concat", humanObserved_Features_Concat.shape)

humanObserved_Target_Sub = humanObserved_Features_Sub[:, [9]]
humanObserved_Features_Sub = np.delete(humanObserved_Features_Sub, [9], axis=1)
print("humanObserved_Target_Sub", humanObserved_Target_Sub.shape)
print("humanObserved_Features_Sub", humanObserved_Features_Sub.shape)

# Training, Testing and Validation for HumanObserved_Same pairs
Training_humanObserved_Target_Concat = GenerateTrainingTarget(humanObserved_Target_Concat, TrainingPercent)
Training_humanObserved_Features_Concat = GenerateTrainingDataMatrix(humanObserved_Features_Concat, TrainingPercent)
print("Training_humanObserved_Target_Concat: ", Training_humanObserved_Target_Concat.shape)
print("Training_humanObserved_Features_Concat :", Training_humanObserved_Features_Concat.shape)

Val_humanObserved_Target_Concat = GenerateValTargetVector(humanObserved_Target_Concat, ValidationPercent,
                                                          (len(Training_humanObserved_Target_Concat)))
Val_humanObserved_Features_Concat = GenerateValData(humanObserved_Features_Concat, ValidationPercent,
                                                    (len(Training_humanObserved_Target_Concat)))
print("Val_humanObserved_Target_Concat: ", Val_humanObserved_Target_Concat.shape)
print("Val_humanObserved_Features_Concat: ", Val_humanObserved_Features_Concat.shape)

Test_humanObserved_Target_Concat = GenerateValTargetVector(humanObserved_Target_Concat, TestPercent, (
        len(Training_humanObserved_Target_Concat) + len(Val_humanObserved_Target_Concat)))
Test_humanObserved_Features_Concat = GenerateValData(humanObserved_Features_Concat, TestPercent, (
        len(Training_humanObserved_Target_Concat) + len(Val_humanObserved_Target_Concat)))
print("Test_humanObserved_Target_Concat:", Test_humanObserved_Target_Concat.shape)
print("Test_humanObserved_Features_Concat:", Test_humanObserved_Features_Concat.shape)

# Training, Testing and Validation for HumanObserved_Diff pairs
Training_humanObserved_Target_Sub = GenerateTrainingTarget(humanObserved_Target_Sub, TrainingPercent)
Training_humanObserved_Features_Sub = GenerateTrainingDataMatrix(humanObserved_Features_Sub, TrainingPercent)
print("Training_humanObserved_Target_Sub: ", Training_humanObserved_Target_Sub.shape)
print("Training_humanObserved_Features_Sub :", Training_humanObserved_Features_Sub.shape)

Val_humanObserved_Target_Sub = GenerateValTargetVector(humanObserved_Target_Sub, ValidationPercent,
                                                       (len(Training_humanObserved_Target_Sub)))
Val_humanObserved_Features_Sub = GenerateValData(humanObserved_Features_Sub, ValidationPercent,
                                                 (len(Training_humanObserved_Target_Sub)))
print("Val_humanObserved_Target_Sub: ", Val_humanObserved_Target_Sub.shape)
print("Val_humanObserved_Features_Sub: ", Val_humanObserved_Features_Sub.shape)

Test_humanObserved_Target_Sub = GenerateValTargetVector(humanObserved_Target_Sub, TestPercent, (
        len(Training_humanObserved_Target_Sub) + len(Val_humanObserved_Target_Sub)))
Test_humanObserved_Features_Sub = GenerateValData(humanObserved_Features_Sub, TestPercent, (
        len(Training_humanObserved_Target_Sub) + len(Val_humanObserved_Target_Sub)))
print("Test_humanObserved_Target_Sub:", Test_humanObserved_Target_Sub.shape)
print("Test_humanObserved_Features_Sub:", Test_humanObserved_Features_Sub.shape)

print("--------------------------------------------")
# ================== GSC ========================== #
GSCObserved = readFile(path.relpath("GSC-Features-Data/GSC-Features.csv"))
GSC_SamePairs = readFile(path.relpath("GSC-Features-Data/same_pairs.csv"))
GSC_SamePairs = GSC_SamePairs.sample(frac=1).reset_index(drop=True)
GSC_SamePairs = GSC_SamePairs.head(3000)
GSC_DiffPairs = readFile(path.relpath("GSC-Features-Data/diffn_pairs.csv"))
GSC_DiffPairs = GSC_DiffPairs.sample(frac=1).reset_index(drop=True)
GSC_DiffPairs = GSC_DiffPairs.head(3000)
GSCObserved_Same_Features = getFeatures(GSCObserved, GSC_SamePairs)
GSCObserved_Features_Concat = GSCObserved_Same_Features[0]
GSCObserved_Same_Features_Sub = GSCObserved_Same_Features[1]
GSCObserved_Target_Concat = np.ones(shape=(len(GSCObserved_Features_Concat), 1))

GSCObserved_Diff_Features = getFeatures(GSCObserved, GSC_DiffPairs)
GSCObserved_Features_Sub = GSCObserved_Diff_Features[0]
GSCObserved_Diff_Features_Sub = GSCObserved_Diff_Features[1]
GSCObserved_Target_Sub = np.zeros(shape=(len(GSCObserved_Features_Sub), 1))

GSCObserved_Features_Concat = np.append(GSCObserved_Features_Concat, GSCObserved_Target_Concat, axis=1)
GSCObserved_Same_Features_Sub = np.append(GSCObserved_Same_Features_Sub, GSCObserved_Target_Concat, axis=1)
GSCObserved_Features_Sub = np.append(GSCObserved_Features_Sub, GSCObserved_Target_Sub, axis=1)
GSCObserved_Diff_Features_Sub = np.append(GSCObserved_Diff_Features_Sub, GSCObserved_Target_Sub, axis=1)

GSCObserved_Features_Concat = np.append(GSCObserved_Features_Concat, GSCObserved_Features_Sub, axis=0)
GSCObserved_Features_Sub = np.append(GSCObserved_Same_Features_Sub, GSCObserved_Diff_Features_Sub, axis=0)

np.random.shuffle(GSCObserved_Features_Concat)
np.random.shuffle(GSCObserved_Features_Sub)

GSCObserved_Features_Concat, GSCObserved_Features_Concat_length = generate_raw_data_with_relevant_features(
    GSCObserved_Features_Concat, \
    is_synthetic=False)
GSCObserved_Features_Sub, GSCObserved_Features_Sub_length = generate_raw_data_with_relevant_features(
    GSCObserved_Features_Sub, \
    is_synthetic=False)
GSCObserved_Target_Concat = GSCObserved_Features_Concat[:, [GSCObserved_Features_Concat_length - 1]]
GSCObserved_Features_Concat = np.delete(GSCObserved_Features_Concat, [GSCObserved_Features_Concat_length - 1], axis=1)
print("GSCObserved_Target_Concat: ", GSCObserved_Target_Concat.shape)
print("GSCObserved_Features_Concat: ", GSCObserved_Features_Concat.shape)

GSCObserved_Target_Sub = GSCObserved_Features_Sub[:, [GSCObserved_Features_Sub_length - 1]]
GSCObserved_Features_Sub = np.delete(GSCObserved_Features_Sub, [GSCObserved_Features_Sub_length - 1], axis=1)
print("GSCObserved_Target_Sub: ", GSCObserved_Target_Sub.shape)
print("GSCObserved_Features_Sub: ", GSCObserved_Features_Sub.shape)

# Training, testing and validation for GSC_Same pairs
Training_GSCObserved_Target_Concat = GenerateTrainingTarget(GSCObserved_Target_Concat, TrainingPercent)
Training_GSCObserved_Features_Concat = GenerateTrainingDataMatrix(GSCObserved_Features_Concat, TrainingPercent)
print("Training_GSCObserved_Target_Concat: ", Training_GSCObserved_Target_Concat.shape)
print("Training_GSCObserved_Features_Concat :", Training_GSCObserved_Features_Concat.shape)

Val_GSCObserved_Target_Concat = GenerateValTargetVector(GSCObserved_Target_Concat, ValidationPercent,
                                                        (len(Training_GSCObserved_Target_Concat)))
Val_GSCObserved_Features_Concat = GenerateValData(GSCObserved_Features_Concat, ValidationPercent,
                                                  (len(Training_GSCObserved_Target_Concat)))
print("Val_GSCObserved_Target_Concat: ", Val_GSCObserved_Target_Concat.shape)
print("Val_GSCObserved_Features_Concat: ", Val_GSCObserved_Features_Concat.shape)

Test_GSCObserved_Target_Concat = GenerateValTargetVector(GSCObserved_Target_Concat, TestPercent, (
        len(Training_GSCObserved_Target_Concat) + len(Val_GSCObserved_Target_Concat)))
Test_GSCObserved_Features_Concat = GenerateValData(GSCObserved_Features_Concat, TestPercent, (
        len(Training_GSCObserved_Target_Concat) + len(Val_GSCObserved_Target_Concat)))
print("Test_GSCObserved_Target_Concat:", Test_GSCObserved_Target_Concat.shape)
print("Test_GSCObserved_Features_Concat:", Test_GSCObserved_Features_Concat.shape)

# Training, testing and Validation for GSC_different pairs
Training_GSCObserved_Target_Sub = GenerateTrainingTarget(GSCObserved_Target_Sub, TrainingPercent)
Training_GSCObserved_Features_Sub = GenerateTrainingDataMatrix(GSCObserved_Features_Sub, TrainingPercent)
print("Training_GSCObserved_Diff_Target: ", Training_GSCObserved_Target_Sub.shape)
print("Training_GSCObserved_Diff_Features :", Training_GSCObserved_Features_Sub.shape)

Val_GSCObserved_Target_Sub = GenerateValTargetVector(GSCObserved_Target_Sub, ValidationPercent,
                                                     (len(Training_GSCObserved_Target_Sub)))
Val_GSCObserved_Features_Sub = GenerateValData(GSCObserved_Features_Sub, ValidationPercent,
                                               (len(Training_GSCObserved_Target_Sub)))
print("Val_GSCObserved_Diff_Target: ", Val_GSCObserved_Target_Sub.shape)
print("Val_GSCObserved_Diff_Features: ", Val_GSCObserved_Features_Sub.shape)

Test_GSCObserved_Target_Sub = GenerateValTargetVector(GSCObserved_Target_Sub, TestPercent, (
        len(Training_GSCObserved_Target_Sub) + len(Val_GSCObserved_Target_Sub)))
Test_GSCObserved_Features_Sub = GenerateValData(GSCObserved_Features_Sub, TestPercent, (
        len(Training_GSCObserved_Target_Sub) + len(Val_GSCObserved_Target_Sub)))
print("Test_GSCObserved_Diff_Target:", Test_GSCObserved_Target_Sub.shape)
print("Test_GSCObserved_Diff_Features:", Test_GSCObserved_Features_Sub.shape)


def generate_big_sigma_matrix(data, number_of_features, is_synthetic):
    big_sigma_matrix = np.zeros(shape=(number_of_features, number_of_features))
    variance_matrix = np.var(data, axis=0)
    for feature in range(number_of_features):
        big_sigma_matrix[feature][feature] = variance_matrix[feature]
    if is_synthetic:
        big_sigma_matrix = np.dot(3, big_sigma_matrix)
    else:
        big_sigma_matrix = np.dot(200, big_sigma_matrix)
    return big_sigma_matrix


def generate_phi_matrix(input_values, mu, big_sigma_inverse):
    phi_matrix = np.zeros((len(input_values), len(mu)))
    for feature in range(0, len(mu)):
        for input_index in range(0, len(input_values)):
            phi_matrix[input_index][feature] = GetRadialBasisOut(
                input_values[input_index], mu[feature], big_sigma_inverse)
    return phi_matrix


def GetRadialBasisOut(input_vector, mu_vector, big_sigma_inverse):
    return np.exp(-0.5 * np.dot(np.dot(np.subtract(input_vector, mu_vector),
                                       big_sigma_inverse), np.transpose(np.subtract(input_vector, mu_vector))))


def compute_rms_error_and_accuracy(computed_output, actual_output):
    total_error = 0
    counter = 0
    for i in range(0, len(computed_output)):
        total_error += math.pow(computed_output[i] - actual_output[i], 2)
        if int(np.around(computed_output[i], 0)) == actual_output[i]:
            counter += 1
    accuracy = (float((counter * 100)) / float(len(computed_output)))
    return math.sqrt(total_error / len(computed_output)), accuracy


def compute_output(weight_matrix, phi_matrix):
    return np.dot(phi_matrix, weight_matrix)


Erms_Val = []
Erms_TR = []
Erms_Test = []

Acc_Val = []
Acc_TR = []
Acc_Test = []

learningRate = [0.01, 0.03, 0.05, 0.07]
C_Lambda = [0.01, 0.03, 0.06, 0.09]
M = [10, 60, 100, 120, 150]
human_concat = 18
human_sub = 9


def clear_list():
    Erms_Val.clear()
    Erms_TR.clear()
    Erms_Test.clear()
    Acc_Val.clear()
    Acc_TR.clear()
    Acc_Test.clear()


# gradient for human observed training concat
def linear_regression_solution(training_phi_matrix, training_target, M, LR, lambda_value):
    number_of_clusters = M
    lambda_term = lambda_value
    learning_rate = LR
    weight_matrix = np.zeros((number_of_clusters, 1))
    for i in range(NUMBER_OF_ITERATIONS):
        predicted_output = np.dot(training_phi_matrix, weight_matrix)
        weight_regularized = np.dot(lambda_term, weight_matrix)
        calculated_error = np.subtract(predicted_output, training_target)
        calculated_gradient = (1 / len(training_phi_matrix)) * np.dot(np.transpose(calculated_error),
                                                                      training_phi_matrix)
        weight_regularized = calculated_gradient.transpose() + weight_regularized
        weight_matrix = np.subtract(weight_matrix, learning_rate * weight_regularized)
    return weight_matrix


def linear_regression_human_concat(M, LR, lambda_value):
    concat_big_sigma = generate_big_sigma_matrix(humanObserved_Features_Concat, number_of_features=human_concat,
                                                 is_synthetic=False)
    concat_big_sigma_inverse = np.linalg.inv(concat_big_sigma)
    Training_Human_Concat_Mean = KMeans(n_clusters=M, random_state=0).fit(Training_humanObserved_Features_Concat)
    mu_matrix_training = Training_Human_Concat_Mean.cluster_centers_
    training_human_phi_matrix = generate_phi_matrix(Training_humanObserved_Features_Concat, mu_matrix_training,
                                                    concat_big_sigma_inverse)
    testing_human_phi_matrix = generate_phi_matrix(Test_humanObserved_Features_Concat, mu_matrix_training,
                                                   concat_big_sigma_inverse)
    validation_human_phi_matrix = generate_phi_matrix(Val_humanObserved_Features_Concat, mu_matrix_training,
                                                      concat_big_sigma_inverse)

    concat_human_weight_matrix = linear_regression_solution(training_human_phi_matrix,
                                                            Training_humanObserved_Target_Concat, M, LR, lambda_value)

    training_output = compute_output(concat_human_weight_matrix, training_human_phi_matrix)
    validation_output = compute_output(concat_human_weight_matrix, validation_human_phi_matrix)
    testing_output = compute_output(concat_human_weight_matrix, testing_human_phi_matrix)

    return training_output, validation_output, testing_output


def linear_regression_human_sub(M, LR, lambda_value):
    sub_big_sigma = generate_big_sigma_matrix(humanObserved_Features_Sub, number_of_features=human_sub,
                                              is_synthetic=False)
    concat_big_sigma_inverse = np.linalg.inv(sub_big_sigma)
    Training_Human_Concat_Mean = KMeans(n_clusters=M, random_state=0).fit(humanObserved_Features_Sub)
    mu_matrix_training = Training_Human_Concat_Mean.cluster_centers_
    training_human_phi_matrix = generate_phi_matrix(Training_humanObserved_Features_Sub, mu_matrix_training,
                                                    concat_big_sigma_inverse)
    testing_human_phi_matrix = generate_phi_matrix(Test_humanObserved_Features_Sub, mu_matrix_training,
                                                   concat_big_sigma_inverse)
    validation_human_phi_matrix = generate_phi_matrix(Val_humanObserved_Features_Sub, mu_matrix_training,
                                                      concat_big_sigma_inverse)

    concat_human_weight_matrix = linear_regression_solution(training_human_phi_matrix,
                                                            Training_humanObserved_Target_Sub, M, LR, lambda_value)

    training_output = compute_output(concat_human_weight_matrix, training_human_phi_matrix)
    validation_output = compute_output(concat_human_weight_matrix, validation_human_phi_matrix)
    testing_output = compute_output(concat_human_weight_matrix, testing_human_phi_matrix)

    return training_output, validation_output, testing_output


def linear_regression_gsc_concat(M, LR, lambda_value):
    concat_big_sigma = generate_big_sigma_matrix(GSCObserved_Features_Concat,
                                                 number_of_features=GSCObserved_Features_Concat_length - 1,
                                                 is_synthetic=False)
    concat_big_sigma_inverse = np.linalg.inv(concat_big_sigma)
    Training_GSC_Concat_Mean = KMeans(n_clusters=M, random_state=0).fit(GSCObserved_Features_Concat)
    mu_matrix_training = Training_GSC_Concat_Mean.cluster_centers_
    training_GSC_phi_matrix = generate_phi_matrix(Training_GSCObserved_Features_Concat, mu_matrix_training,
                                                  concat_big_sigma_inverse)
    testing_GSC_phi_matrix = generate_phi_matrix(Test_GSCObserved_Features_Concat, mu_matrix_training,
                                                 concat_big_sigma_inverse)
    validation_GSC_phi_matrix = generate_phi_matrix(Val_GSCObserved_Features_Concat, mu_matrix_training,
                                                    concat_big_sigma_inverse)

    concat_human_weight_matrix = linear_regression_solution(training_GSC_phi_matrix, Training_GSCObserved_Target_Concat,
                                                            M, LR, lambda_value)

    training_output = compute_output(concat_human_weight_matrix, training_GSC_phi_matrix)
    validation_output = compute_output(concat_human_weight_matrix, validation_GSC_phi_matrix)
    testing_output = compute_output(concat_human_weight_matrix, testing_GSC_phi_matrix)

    return training_output, validation_output, testing_output


def linear_regression_gsc_sub(M, LR, lambda_value):
    sub_big_sigma = generate_big_sigma_matrix(GSCObserved_Features_Sub,
                                              number_of_features=GSCObserved_Features_Sub_length - 1,
                                              is_synthetic=False)
    sub_big_sigma_inverse = np.linalg.inv(sub_big_sigma)
    Training_GSC_sub_Mean = KMeans(n_clusters=M, random_state=0).fit(GSCObserved_Features_Sub)
    mu_matrix_training = Training_GSC_sub_Mean.cluster_centers_
    training_gsc_phi_matrix = generate_phi_matrix(Training_GSCObserved_Features_Sub, mu_matrix_training,
                                                  sub_big_sigma_inverse)
    testing_gsc_phi_matrix = generate_phi_matrix(Test_GSCObserved_Features_Sub, mu_matrix_training,
                                                 sub_big_sigma_inverse)
    validation_gsc_phi_matrix = generate_phi_matrix(Val_GSCObserved_Features_Sub, mu_matrix_training,
                                                    sub_big_sigma_inverse)

    concat_human_weight_matrix = linear_regression_solution(training_gsc_phi_matrix, Training_GSCObserved_Target_Sub, M,
                                                            LR, lambda_value)

    training_output = compute_output(concat_human_weight_matrix, training_gsc_phi_matrix)
    validation_output = compute_output(concat_human_weight_matrix, validation_gsc_phi_matrix)
    testing_output = compute_output(concat_human_weight_matrix, testing_gsc_phi_matrix)

    return training_output, validation_output, testing_output


def plot_graph_learning():
    plt.figure(figsize=[6, 6])
    plt.plot(learningRate, Erms_TR, label="Training")
    plt.plot(learningRate, Erms_Val, label="Validation")
    plt.plot(learningRate, Erms_Test, label="Testing")
    plt.legend()
    plt.xlabel('Learning Rate')
    plt.ylabel('ERMS')
    plt.title('Learning rate VS ERMS')
    plt.show()

    plt.figure(figsize=[6, 6])
    plt.plot(learningRate, Acc_TR, label="Training")
    plt.plot(learningRate, Acc_Val, label="Validation")
    plt.plot(learningRate, Acc_Test, label="Testing")
    plt.legend()
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Learning rate VS Accuracy')
    plt.show()


def plot_graph_M():
    plt.figure(figsize=[6, 6])
    plt.plot(M, Erms_TR, label="Training")
    plt.plot(M, Erms_Val, label="Validation")
    plt.plot(M, Erms_Test, label="Testing")
    plt.legend()
    plt.xlabel('Number of clusters')
    plt.ylabel('ERMS')
    plt.title('Number of Clusters VS ERMS')
    plt.show()

    plt.figure(figsize=[6, 6])
    plt.plot(M, Acc_TR, label="Training")
    plt.plot(M, Acc_Val, label="Validation")
    plt.plot(M, Acc_Test, label="Testing")
    plt.legend()
    plt.xlabel('Number of clusters')
    plt.ylabel('Accuracy')
    plt.title('Number of clusters VS Accuracy')
    plt.show()


def plot_graph_Lambda():
    plt.figure(figsize=[6, 6])
    plt.plot(C_Lambda, Erms_TR, label="Training")
    plt.plot(C_Lambda, Erms_Val, label="Validation")
    plt.plot(C_Lambda, Erms_Test, label="Testing")
    plt.legend()
    plt.xlabel('Lambda')
    plt.ylabel('ERMS')
    plt.title('Lambda VS ERMS')
    plt.show()

    plt.figure(figsize=[6, 6])
    plt.plot(C_Lambda, Acc_TR, label="Training")
    plt.plot(C_Lambda, Acc_Val, label="Validation")
    plt.plot(C_Lambda, Acc_Test, label="Testing")
    plt.legend()
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Lambda VS Accuracy')
    plt.show()

def printAccuracy():
    print("E_rms Training   = " + str(np.around(min(Erms_TR), 5)))
    print("E_rms Validation = " + str(np.around(min(Erms_Val), 5)))
    print("E_rms Testing    = " + str(np.around(min(Erms_Test), 5)))

    print("ACC Training   = " + str(np.around(max(Acc_TR), 5)))
    print("ACC Validation = " + str(np.around(max(Acc_Val), 5)))
    print("ACC Testing    = " + str(np.around(max(Acc_Test), 5)))

def cal_erms_acc_human_concat_m_values():
    clear_list()
    for i in range(len(M)):
        training_output, validation_output, testing_output = linear_regression_human_concat(M[i], LR=0.01,
                                                                                            lambda_value=2)
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_humanObserved_Target_Concat)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_humanObserved_Target_Concat)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_humanObserved_Target_Concat)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_human_sub_m_values():
    clear_list()
    for i in range(len(M)):
        training_output, validation_output, testing_output = linear_regression_human_sub(M[i], LR=0.01, lambda_value=2)
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_humanObserved_Target_Sub)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_humanObserved_Target_Sub)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_humanObserved_Target_Sub)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_human_concat_LR_values():
    clear_list()
    for i in range(len(learningRate)):
        training_output, validation_output, testing_output = linear_regression_human_concat(10, learningRate[i],
                                                                                            lambda_value=2)
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_humanObserved_Target_Concat)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_humanObserved_Target_Concat)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_humanObserved_Target_Concat)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_human_sub_LR_values():
    clear_list()
    for i in range(len(learningRate)):
        training_output, validation_output, testing_output = linear_regression_human_sub(10, learningRate[i],
                                                                                         lambda_value=2)
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_humanObserved_Target_Sub)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_humanObserved_Target_Sub)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_humanObserved_Target_Sub)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_human_concat_Lambda_values():
    clear_list()
    for i in range(len(C_Lambda)):
        training_output, validation_output, testing_output = linear_regression_human_concat(10, 0.01, C_Lambda[i])
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_humanObserved_Target_Concat)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_humanObserved_Target_Concat)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_humanObserved_Target_Concat)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_human_sub_Lambda_values():
    clear_list()
    for i in range(len(C_Lambda)):
        training_output, validation_output, testing_output = linear_regression_human_sub(10, 0.01, C_Lambda[i])
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_humanObserved_Target_Sub)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_humanObserved_Target_Sub)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_humanObserved_Target_Sub)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_gsc_concat_m_values():
    clear_list()
    for i in range(len(M)):
        training_output, validation_output, testing_output = linear_regression_gsc_concat(M[i], LR=0.01, lambda_value=2)
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_GSCObserved_Target_Concat)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_GSCObserved_Target_Concat)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_GSCObserved_Target_Concat)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_gsc_sub_m_values():
    clear_list()
    for i in range(len(M)):
        training_output, validation_output, testing_output = linear_regression_gsc_sub(M[i], LR=0.01, lambda_value=2)
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_GSCObserved_Target_Sub)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_GSCObserved_Target_Sub)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_GSCObserved_Target_Sub)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_gsc_concat_LR_values():
    clear_list()
    for i in range(len(learningRate)):
        training_output, validation_output, testing_output = linear_regression_gsc_concat(10, learningRate[i],
                                                                                          lambda_value=2)
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_GSCObserved_Target_Concat)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_GSCObserved_Target_Concat)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_GSCObserved_Target_Concat)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_gsc_sub_LR_values():
    clear_list()
    for i in range(len(learningRate)):
        training_output, validation_output, testing_output = linear_regression_gsc_sub(10, learningRate[i],
                                                                                       lambda_value=2)
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_GSCObserved_Target_Sub)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_GSCObserved_Target_Sub)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_GSCObserved_Target_Sub)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_gsc_concat_Lambda_values():
    clear_list()
    for i in range(len(C_Lambda)):
        training_output, validation_output, testing_output = linear_regression_gsc_concat(10, 0.01, C_Lambda[i])
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_GSCObserved_Target_Concat)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_GSCObserved_Target_Concat)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_GSCObserved_Target_Concat)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def cal_erms_acc_gsc_sub_Lambda_values():
    clear_list()
    for i in range(len(C_Lambda)):
        training_output, validation_output, testing_output = linear_regression_gsc_sub(10, 0.01, C_Lambda[i])
        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_GSCObserved_Target_Sub)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_GSCObserved_Target_Sub)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_GSCObserved_Target_Sub)
        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


# -------------- HumanObserverd -------------- #
# cal_erms_acc_human_concat_m_values()
# plot_graph_M()
# print("Linear regression human observed concat:")
# printAccuracy()
#
# cal_erms_acc_human_sub_m_values()
# plot_graph_M()
# print("Linear regression human observed sub:")
# printAccuracy()
#
# cal_erms_acc_human_concat_LR_values()
# plot_graph_learning()
#
# cal_erms_acc_human_sub_LR_values()
# plot_graph_learning()
#
# cal_erms_acc_human_sub_Lambda_values()
# plot_graph_Lambda()
#
# cal_erms_acc_human_concat_Lambda_values()
# plot_graph_Lambda()

# ------------------ GSC -----------------------#
cal_erms_acc_gsc_concat_m_values()
plot_graph_M()
print("Linear regression GSC observed concat:")
printAccuracy()
cal_erms_acc_gsc_concat_LR_values()
plot_graph_learning()

cal_erms_acc_gsc_sub_LR_values()
plot_graph_learning()
print("Linear regression GSC observed sub:")
printAccuracy()

cal_erms_acc_gsc_sub_Lambda_values()
plot_graph_Lambda()

cal_erms_acc_gsc_concat_Lambda_values()
plot_graph_Lambda()

############################## Logistic Regression ###########################
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logestic_regression_solution(training_input_matrix, training_target, LR, lambda_value):
    lambda_term = lambda_value
    learning_rate = LR
    weight_matrix = np.zeros((training_input_matrix.shape[1], 1))
    for i in range(NUMBER_OF_ITERATIONS):
        predicted_output = np.dot(training_input_matrix, weight_matrix)
        predicted_output = sigmoid(predicted_output)
        predicted_output = predicted_output >= 0.5
        predicted_output = predicted_output.astype(int)
        weight_regularized = np.dot(lambda_term, weight_matrix)
        calculated_error = np.subtract(predicted_output, training_target)
        calculated_gradient = (1 / len(training_input_matrix)) * np.dot(np.transpose(calculated_error),
                                                                        training_input_matrix)
        weight_regularized = calculated_gradient.transpose() + weight_regularized
        weight_matrix = np.subtract(weight_matrix, learning_rate * weight_regularized)
    return weight_matrix


# logestic_regression_solution(Training_humanObserved_Features_Concat,Training_humanObserved_Target_Concat,0.01,2)
def logestic_cal_erms_acc_human_concat_LR_values():
    clear_list()
    for i in range(len(learningRate)):
        weight_matrix = logestic_regression_solution(Training_humanObserved_Features_Concat,
                                                     Training_humanObserved_Target_Concat, learningRate[i], 2)
        training_output = compute_output(weight_matrix, Training_humanObserved_Features_Concat)
        validation_output = compute_output(weight_matrix, Val_humanObserved_Features_Concat)
        testing_output = compute_output(weight_matrix, Test_humanObserved_Features_Concat)

        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_humanObserved_Target_Concat)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_humanObserved_Target_Concat)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_humanObserved_Target_Concat)

        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def logestic_cal_erms_acc_human_Sub_LR_values():
    clear_list()
    for i in range(len(learningRate)):
        weight_matrix = logestic_regression_solution(Training_humanObserved_Features_Sub,
                                                     Training_humanObserved_Target_Sub, learningRate[i], 2)
        training_output = compute_output(weight_matrix, Training_humanObserved_Features_Sub)
        validation_output = compute_output(weight_matrix, Val_humanObserved_Features_Sub)
        testing_output = compute_output(weight_matrix, Test_humanObserved_Features_Sub)

        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_humanObserved_Target_Sub)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_humanObserved_Target_Sub)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_humanObserved_Target_Sub)

        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def logestic_cal_erms_acc_human_Concat_Lambda_values():
    clear_list()
    for i in range(len(learningRate)):
        weight_matrix = logestic_regression_solution(Training_humanObserved_Features_Concat,
                                                     Training_humanObserved_Target_Concat, 0.01, C_Lambda[i])
        training_output = compute_output(weight_matrix, Training_humanObserved_Features_Concat)
        validation_output = compute_output(weight_matrix, Val_humanObserved_Features_Concat)
        testing_output = compute_output(weight_matrix, Test_humanObserved_Features_Concat)

        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_humanObserved_Target_Concat)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_humanObserved_Target_Concat)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_humanObserved_Target_Concat)

        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def logestic_cal_erms_acc_human_Sub_Lambda_values():
    clear_list()
    for i in range(len(learningRate)):
        weight_matrix = logestic_regression_solution(Training_humanObserved_Features_Sub,
                                                     Training_humanObserved_Target_Sub, 0.01, C_Lambda[i])
        training_output = compute_output(weight_matrix, Training_humanObserved_Features_Sub)
        validation_output = compute_output(weight_matrix, Val_humanObserved_Features_Sub)
        testing_output = compute_output(weight_matrix, Test_humanObserved_Features_Sub)

        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_humanObserved_Target_Sub)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_humanObserved_Target_Sub)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_humanObserved_Target_Sub)

        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


####### GSC Logestic regression #######

def logestic_cal_erms_acc_gsc_concat_LR_values():
    clear_list()
    for i in range(len(learningRate)):
        weight_matrix = logestic_regression_solution(Training_GSCObserved_Features_Concat,
                                                     Training_GSCObserved_Target_Concat, learningRate[i], 2)
        training_output = compute_output(weight_matrix, Training_GSCObserved_Features_Concat)
        validation_output = compute_output(weight_matrix, Val_GSCObserved_Features_Concat)
        testing_output = compute_output(weight_matrix, Test_GSCObserved_Features_Concat)

        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_GSCObserved_Target_Concat)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_GSCObserved_Target_Concat)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_GSCObserved_Target_Concat)

        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def logestic_cal_erms_acc_gsc_Sub_LR_values():
    clear_list()
    for i in range(len(learningRate)):
        weight_matrix = logestic_regression_solution(Training_GSCObserved_Features_Sub,
                                                     Training_GSCObserved_Target_Sub, learningRate[i], 2)
        training_output = compute_output(weight_matrix, Training_GSCObserved_Features_Sub)
        validation_output = compute_output(weight_matrix, Val_GSCObserved_Features_Sub)
        testing_output = compute_output(weight_matrix, Test_GSCObserved_Features_Sub)

        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_GSCObserved_Target_Sub)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_GSCObserved_Target_Sub)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_GSCObserved_Target_Sub)

        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def logestic_cal_erms_acc_gsc_Concat_Lambda_values():
    clear_list()
    for i in range(len(learningRate)):
        weight_matrix = logestic_regression_solution(Training_GSCObserved_Features_Concat,
                                                     Training_GSCObserved_Target_Concat, 0.01, C_Lambda[i])
        training_output = compute_output(weight_matrix, Training_GSCObserved_Features_Concat)
        validation_output = compute_output(weight_matrix, Val_GSCObserved_Features_Concat)
        testing_output = compute_output(weight_matrix, Test_GSCObserved_Features_Concat)

        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_GSCObserved_Target_Concat)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_GSCObserved_Target_Concat)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_GSCObserved_Target_Concat)

        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)


def logestic_cal_erms_acc_gsc_Sub_Lambda_values():
    clear_list()
    for i in range(len(learningRate)):
        weight_matrix = logestic_regression_solution(Training_GSCObserved_Features_Sub,
                                                     Training_GSCObserved_Target_Sub, 0.01, C_Lambda[i])
        training_output = compute_output(weight_matrix, Training_GSCObserved_Features_Sub)
        validation_output = compute_output(weight_matrix, Val_GSCObserved_Features_Sub)
        testing_output = compute_output(weight_matrix, Test_GSCObserved_Features_Sub)

        training_erms, training_accuracy = compute_rms_error_and_accuracy(training_output,
                                                                          Training_GSCObserved_Target_Sub)
        validation_erms, validation_accuracy = compute_rms_error_and_accuracy(validation_output,
                                                                              Val_GSCObserved_Target_Sub)
        testing_erms, testing_accuracy = compute_rms_error_and_accuracy(testing_output,
                                                                        Test_GSCObserved_Target_Sub)

        Erms_TR.append(training_erms)
        Erms_Test.append(testing_erms)
        Erms_Val.append(validation_erms)

        Acc_TR.append(training_accuracy)
        Acc_Val.append(validation_accuracy)
        Acc_Test.append(testing_accuracy)

# -------------- HumanObserverd -------------- #
logestic_cal_erms_acc_human_concat_LR_values()
plot_graph_learning()
print("Logestic regression human observed concat:")
printAccuracy()
logestic_cal_erms_acc_human_Sub_LR_values()
plot_graph_learning()
print("Logestic regression human observed sub:")
printAccuracy()
logestic_cal_erms_acc_human_Concat_Lambda_values()
plot_graph_Lambda()

logestic_cal_erms_acc_human_Sub_Lambda_values()
plot_graph_Lambda()

# ------------------ GSC -----------------------
logestic_cal_erms_acc_gsc_concat_LR_values()
plot_graph_learning()
print("Logestic regression GSC observed concat:")
printAccuracy()
logestic_cal_erms_acc_gsc_Sub_LR_values()
plot_graph_learning()
print("Logestic regression GSC observed sub:")
printAccuracy()
logestic_cal_erms_acc_gsc_Concat_Lambda_values()
plot_graph_Lambda()

logestic_cal_erms_acc_gsc_Sub_Lambda_values()
plot_graph_Lambda()

# -- Neural Network implementation --

# drop_out = 0.1
# first_dense_layer_nodes = 512
# second_dense_layer_nodes = 512
# output_dense_layer_nodes = 2
# human_inputSize_concat = 18
# human_inputSize_sub = 9
# gsc_inputSize_concat = 1014
# gsc_inputSize_sub = 504
#
# def get_model(inputSize):
#     model = Sequential()
#     model.add(Dense(first_dense_layer_nodes, input_dim=(inputSize)))
#     model.add(Activation(
#         'sigmoid'))  # Rectified linear Unit. Unlike sigmoid function, it doesn't sequeeze the the number down. Instead it
#     # focuses on showing the neuron as active or inactive
#
#     # Why dropout?
#     model.add(Dropout(drop_out))
#
#     model.add(Dense(second_dense_layer_nodes))
#     model.add(Activation('sigmoid'))
#
#     model.add(Dense(output_dense_layer_nodes))
#     model.add(Activation('softmax'))
#
#     model.compile(optimizer='rmsprop',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
#
# def process(training_input, training_target):
#     validation_data_split = 0.25
#     num_epochs = 200
#     model_batch_size = 128
#     tb_batch_size = 32
#     early_patience = 100
#
#     training_target = np_utils.to_categorical(np.array(training_target), 2)
#     model = get_model(training_input[1].size)
#     tensorboard_cb = TensorBoard(log_dir='logs', batch_size=tb_batch_size, write_graph=True)
#     earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=0, patience=early_patience, mode='min')
#
#     history = model.fit(training_input
#                         , training_target
#                         , validation_split=validation_data_split
#                         , epochs=num_epochs
#                         , batch_size=model_batch_size
#                         , callbacks=[tensorboard_cb, earlystopping_cb]
#                         )
#
#     return model, history
#
# def validate(model, processedTestData, processedTestLabel):
#     wrong = 0
#     right = 0
#     for i, j in zip(processedTestData, processedTestLabel):
#         y = model.predict(np.array(i).reshape(-1, processedTestData[1].size))
#
#         if j.argmax() == y.argmax():
#             right = right + 1
#         else:
#             wrong = wrong + 1
#
#     print("Errors: " + str(wrong), " Correct :" + str(right))
#
#     print("Testing Accuracy: " + str(right / (right + wrong) * 100))
#
#
# def neural_network(training_feature, val_feature, training_target, val_target, test_feature, test_target):
#     training_human_input = np.append(training_feature, val_feature, axis=0)
#     training_human_target = np.append(training_target, val_target)
#     model, history = process(training_feature[1].size, training_human_input, training_human_target)
#
#     df = pd.DataFrame(history.history)
#     df.plot(subplots=True, grid=True, figsize=(10, 15))
#     plt.show()
#
#     validate(model, test_feature, test_target)
#
#
# print("For Human Concat: ")
# neural_network(human_inputSize_concat,Training_humanObserved_Features_Concat, Val_humanObserved_Features_Concat,
#              Training_humanObserved_Target_Concat, Val_humanObserved_Target_Concat, Test_humanObserved_Features_Concat,
#              Test_humanObserved_Target_Concat)
#
# print("For Human Sub: ")
# neural_network(Training_humanObserved_Features_Sub, Val_humanObserved_Features_Sub,
#              Training_humanObserved_Target_Sub, Val_humanObserved_Target_Sub, Test_humanObserved_Features_Sub,
#              Test_humanObserved_Target_Sub)
#
# print("For GSC Concat: ")
# neural_network(Training_GSCObserved_Features_Concat, Val_GSCObserved_Features_Concat,
#              Training_GSCObserved_Target_Concat, Val_GSCObserved_Target_Concat, Test_GSCObserved_Features_Concat,
#              Test_GSCObserved_Target_Concat)
#
# print("For GSC Sub: ")
# neural_network(Training_GSCObserved_Features_Sub, Val_GSCObserved_Features_Sub,
#              Training_GSCObserved_Target_Sub, Val_GSCObserved_Target_Sub, Test_GSCObserved_Features_Sub,
#              Test_GSCObserved_Target_Sub)