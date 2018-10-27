import collections
import math

import pandas as pd
import numpy as np
from os import path

TrainingPercent = 80  # dividing training data into 80% of the total
ValidationPercent = 10  # validation data is 10% of the total
TestPercent = 10  # testing data is 10 % of the total

def readFile(fileName):
    readLine = pd.read_csv(fileName)
    return readLine

def getFeatures(readData,pairs):
    concatArray =[]
    subArray = []
    target = []
    for index, row in pairs.iterrows():
        img_id1 = readData.loc[readData['img_id'] == row['img_id_A']]
        img_id1 = img_id1.loc[:,'f1':]
        img_id2 = readData.loc[readData['img_id'] == row['img_id_B']]
        img_id2 = img_id2.loc[:, 'f1':]
        concatFeatures = np.concatenate((img_id1.values[0], img_id2.values[0]), axis=0)
        subFeatures = np.absolute(img_id1.values[0]-img_id2.values[0])
        concatArray.append(concatFeatures)
        subArray.append(subFeatures)
    return np.array(concatArray),np.array(subArray)

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData)*0.01*TrainingPercent))
    d2 = rawData[0:T_len,:]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t = rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

humanObserved = readFile(path.relpath("HumanObserved-Features-Data/HumanObserved-Features-Data.csv"))
human_SamePairs = readFile(path.relpath("HumanObserved-Features-Data/same_pairs.csv"))
human_DiffPairs = readFile(path.relpath("HumanObserved-Features-Data/diffn_pairs.csv"))
human_DiffPairs = human_DiffPairs.sample(frac=1).reset_index(drop=True)
human_DiffPairs = human_DiffPairs.head(791)

# Human Observed Same Pairs.
humanObserved_Same_Features = getFeatures(humanObserved, human_SamePairs)
humanObserved_Same_Features_Concat = humanObserved_Same_Features[0]
humanObserved_Same_Features_Sub = humanObserved_Same_Features[1]
humanObserved_Same_Target = np.ones(shape=(len(humanObserved_Same_Features_Concat), 1))

# Human Observed different Pairs.
humanObserved_Diff_Features = getFeatures(humanObserved, human_DiffPairs)
humanObserved_Diff_Features_Concat = humanObserved_Diff_Features[0]
humanObserved_Diff_Features_Sub = humanObserved_Diff_Features[1]
humanObserved_Diff_Target = np.zeros(shape=(len(humanObserved_Diff_Features_Concat), 1))

humanObserved_Same_Features_Concat = np.append(humanObserved_Same_Features_Concat,humanObserved_Same_Target,axis=1)
humanObserved_Same_Features_Sub = np.append(humanObserved_Same_Features_Sub,humanObserved_Same_Target,axis=1)
humanObserved_Diff_Features_Concat = np.append(humanObserved_Diff_Features_Concat,humanObserved_Diff_Target,axis=1)
humanObserved_Diff_Features_Sub = np.append(humanObserved_Diff_Features_Sub,humanObserved_Diff_Target,axis=1)

humanObserved_Features_Concat = np.append(humanObserved_Same_Features_Concat, humanObserved_Diff_Features_Concat, axis=0)
humanObserved_Features_Sub = np.append(humanObserved_Same_Features_Sub, humanObserved_Diff_Features_Sub, axis=0)

np.random.shuffle(humanObserved_Features_Concat)
np.random.shuffle(humanObserved_Features_Sub)

humanObserved_Same_Target = humanObserved_Features_Concat[:, [18]]
humanObserved_Same_Features_Concat = np.delete(humanObserved_Features_Concat, [18], axis=1)
print("humanObserved_Same_Target",humanObserved_Same_Target.shape)
print("humanObserved_Same_Features_Concat",humanObserved_Same_Features_Concat.shape)
print("humanObserved_Same_Features_Sub",humanObserved_Same_Features_Sub.shape)

humanObserved_Diff_Target = humanObserved_Features_Sub[:, [9]]
humanObserved_Diff_Features_Concat = np.delete(humanObserved_Features_Sub, [9], axis=1)
print("humanObserved_Diff_Target", humanObserved_Diff_Target.shape)
print("humanObserved_Diff_Features_Concat", humanObserved_Diff_Features_Concat.shape)
print("humanObserved_Diff_Features_Sub", humanObserved_Diff_Features_Sub.shape)

# Training, Testing and Validation for HumanObserved_Same pairs
Training_HumanObserved_Same_Target = GenerateTrainingTarget(humanObserved_Same_Target, TrainingPercent)
Training_HumanObserved_Same_Features = GenerateTrainingDataMatrix(humanObserved_Same_Features_Concat, TrainingPercent)
print("TrainingHumanObservedSameTarget: ", Training_HumanObserved_Same_Target.shape)
print("TrainingHumanObservedSameFeatures :", Training_HumanObserved_Same_Features.shape)

Val_HumanObserved_Same_Target = GenerateValTargetVector(humanObserved_Same_Target, ValidationPercent, (len(Training_HumanObserved_Same_Target)))
Val_HumanObserved_Same_Features = GenerateValData(humanObserved_Same_Features_Concat, ValidationPercent, (len(Training_HumanObserved_Same_Target)))
print("ValHumanObservedSameTarget: ", Val_HumanObserved_Same_Target.shape)
print("ValHumanObservedSameFeatures: ", Val_HumanObserved_Same_Features.shape)

Test_HumanObserved_Same_Target = GenerateValTargetVector(humanObserved_Same_Target, TestPercent, (len(Training_HumanObserved_Same_Target) + len(Val_HumanObserved_Same_Target)))
Test_HumanObserved_Same_Features = GenerateValData(humanObserved_Same_Features_Concat, TestPercent, (len(Training_HumanObserved_Same_Target) + len(Val_HumanObserved_Same_Target)))
print("TestHumanObservedSameTarget:", Test_HumanObserved_Same_Target.shape)
print("TestHumanObservedSameFeatures:", Test_HumanObserved_Same_Features.shape)

# Training, Testing and Validation for HumanObserved_Diff pairs
Training_HumanObserved_Diff_Target = GenerateTrainingTarget(humanObserved_Diff_Target, TrainingPercent)
Training_HumanObserved_Diff_Features = GenerateTrainingDataMatrix(humanObserved_Diff_Features_Concat, TrainingPercent)
print("TrainingHumanObservedDiffTarget: ", Training_HumanObserved_Diff_Target.shape)
print("TrainingHumanObservedDiffFeatures :", Training_HumanObserved_Diff_Features.shape)

Val_HumanObserved_Diff_Target = GenerateValTargetVector(humanObserved_Diff_Target, ValidationPercent, (len(Training_HumanObserved_Diff_Target)))
Val_HumanObserved_Diff_Features = GenerateValData(humanObserved_Diff_Features_Concat, ValidationPercent, (len(Training_HumanObserved_Diff_Target)))
print("ValHumanObservedDiffTarget: ", Val_HumanObserved_Diff_Target.shape)
print("ValHumanObservedSDiffFeatures: ", Val_HumanObserved_Diff_Features.shape)

Test_HumanObserved_Diff_Target = GenerateValTargetVector(humanObserved_Diff_Target, TestPercent, (len(Training_HumanObserved_Diff_Target) + len(Val_HumanObserved_Same_Target)))
Test_HumanObserved_Diff_Features = GenerateValData(humanObserved_Diff_Features_Concat, TestPercent, (len(Training_HumanObserved_Diff_Target) + len(Val_HumanObserved_Same_Target)))
print("TestHumanObservedDiffTarget:", Test_HumanObserved_Diff_Target.shape)
print("TestHumanObservedDiffFeatures:", Test_HumanObserved_Diff_Features.shape)


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
GSCObserved_Same_Features_Concat = GSCObserved_Same_Features[0]
GSCObserved_Same_Features_Sub = GSCObserved_Same_Features[1]
GSCObserved_Same_Target = np.ones(shape=(len(GSCObserved_Same_Features_Concat), 1))

GSCObserved_Diff_Features = getFeatures(GSCObserved, GSC_DiffPairs)
GSCObserved_Diff_Features_Concat = GSCObserved_Diff_Features[0]
GSCObserved_Diff_Features_Sub = GSCObserved_Diff_Features[1]
GSCObserved_Diff_Target = np.zeros(shape=(len(GSCObserved_Diff_Features_Concat), 1))

GSCObserved_Same_Features_Concat = np.append(GSCObserved_Same_Features_Concat,GSCObserved_Same_Target,axis=1)
GSCObserved_Same_Features_Sub = np.append(GSCObserved_Same_Features_Sub,GSCObserved_Same_Target,axis=1)
GSCObserved_Diff_Features_Concat = np.append(GSCObserved_Diff_Features_Concat,GSCObserved_Diff_Target,axis=1)
GSCObserved_Diff_Features_Sub = np.append(GSCObserved_Diff_Features_Sub,GSCObserved_Diff_Target,axis=1)

GSCObserved_Features_Concat = np.append(GSCObserved_Same_Features_Concat,GSCObserved_Diff_Features_Concat,axis=0)
GSCObserved_Features_Sub = np.append(GSCObserved_Same_Features_Sub,GSCObserved_Diff_Features_Sub,axis=0)

np.random.shuffle(GSCObserved_Features_Concat)
np.random.shuffle(GSCObserved_Features_Sub)

GSCObserved_Same_Target = GSCObserved_Features_Concat[:,[1024]]
GSCObserved_Same_Features_Concat = np.delete(GSCObserved_Features_Concat,[1024],axis=1)
print("GSCObserved_Same_Target: ", GSCObserved_Same_Target.shape)
print("GSCObserved_Same_Features_Concat: ", GSCObserved_Same_Features_Concat.shape)
print("GSCObserved_Same_Features_Sub: ", GSCObserved_Same_Features_Sub.shape)

GSCObserved_Diff_Target = GSCObserved_Features_Sub[:,[512]]
GSCObserved_Diff_Features_Concat = np.delete(GSCObserved_Features_Sub,[512],axis=1)
print("GSCObserved_Diff_Target: ", GSCObserved_Diff_Target.shape)
print("GSCObserved_Diff_Features_Concat: ", GSCObserved_Diff_Features_Concat.shape)
print("GSCObserved_Diff_Features_Sub: ", GSCObserved_Diff_Features_Sub.shape)

# Training, testing and validation for GSC_Same pairs
Training_GSCObserved_Same_Target = GenerateTrainingTarget(GSCObserved_Same_Target, TrainingPercent)
Training_GSCObserved_Same_Features = GenerateTrainingDataMatrix(GSCObserved_Same_Features_Concat, TrainingPercent)
print("Training_GSCObserved_Same_Target: ", Training_GSCObserved_Same_Target.shape)
print("Training_GSCObserved_Same_Features :", Training_GSCObserved_Same_Features.shape)

Val_GSCObserved_Same_Target = GenerateValTargetVector(GSCObserved_Same_Target, ValidationPercent, (len(Training_GSCObserved_Same_Target)))
Val_GSCObserved_Same_Features = GenerateValData(GSCObserved_Same_Features_Concat, ValidationPercent, (len(Training_GSCObserved_Same_Target)))
print("Val_GSCObserved_Same_Target: ", Val_GSCObserved_Same_Target.shape)
print("Val_GSCObserved_Same_Features: ", Val_GSCObserved_Same_Features.shape)

Test_GSCObserved_Same_Target = GenerateValTargetVector(humanObserved_Same_Target, TestPercent, (len(Training_GSCObserved_Same_Target) + len(Val_GSCObserved_Same_Target)))
Test_GSCObserved_Same_Features = GenerateValData(GSCObserved_Same_Features_Concat, TestPercent, (len(Training_GSCObserved_Same_Target) + len(Val_GSCObserved_Same_Target)))
print("Test_GSCObserved_Same_Target:", Test_GSCObserved_Same_Target.shape)
print("Test_GSCObserved_Same_Features:", Test_GSCObserved_Same_Features.shape)

# Training, testing and Validation for GSC_different pairs
Training_GSCObserved_Diff_Target = GenerateTrainingTarget(GSCObserved_Diff_Target, TrainingPercent)
Training_GSCObserved_Diff_Features = GenerateTrainingDataMatrix(GSCObserved_Diff_Features_Concat, TrainingPercent)
print("Training_GSCObserved_Diff_Target: ", Training_GSCObserved_Diff_Target.shape)
print("Training_GSCObserved_Diff_Features :", Training_GSCObserved_Diff_Features.shape)

Val_GSCObserved_Diff_Target = GenerateValTargetVector(GSCObserved_Diff_Target, ValidationPercent, (len(Training_GSCObserved_Diff_Target)))
Val_GSCObserved_Diff_Features = GenerateValData(GSCObserved_Diff_Features_Concat, ValidationPercent, (len(Training_GSCObserved_Diff_Target)))
print("Val_GSCObserved_Diff_Target: ", Val_GSCObserved_Diff_Target.shape)
print("Val_GSCObserved_Diff_Features: ", Val_GSCObserved_Diff_Features.shape)

Test_GSCObserved_Diff_Target = GenerateValTargetVector(humanObserved_Diff_Target, TestPercent, (len(Training_GSCObserved_Diff_Target) + len(Val_GSCObserved_Diff_Target)))
Test_GSCObserved_Diff_Features = GenerateValData(GSCObserved_Diff_Features_Concat, TestPercent, (len(Training_GSCObserved_Diff_Target) + len(Val_GSCObserved_Diff_Target)))
print("Test_GSCObserved_Diff_Target:", Test_GSCObserved_Diff_Target.shape)
print("Test_GSCObserved_Diff_Features:", Test_GSCObserved_Diff_Features.shape)



