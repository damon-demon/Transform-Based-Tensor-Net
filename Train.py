#       Video Synthesis via Transform-Based Tensor Neural Network
#                             Yimeng Zhang
#                               8/4/2020
#                         yz3397@columbia.edu


import LoadData as LD
import numpy as np
import BuildModel as BM
import TrainModel as TM
import DefineParam as DP
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# 1. Input: Parameters
pixel_w, pixel_h, batchSize, nPhase, nTrainData, nValData, learningRate, nEpoch, nOfModel, ncpkt, trainFile, valFile, testFile, saveDir, modelDir = DP.get_param()


# 2. Data Loading
print('-------------------------------------\nLoading Data...\n-------------------------------------\n')
trainLabel, valLabel = LD.load_train_data(mat73=True)
trainLabel /= 255
valLabel /= 255
trainPhi = np.ones([batchSize, 240, 320, 3])
num_missing = 1
missing_index = [1]
print(missing_index)

for index_x in missing_index:
    trainPhi[:, :, :, index_x] = 0

trainPhi = trainPhi.astype('float32')


# 3. Model Building
print('-------------------------------------\nBuilding Model...\n-------------------------------------\n')
sess, saver, Xinput, Xoutput, Epoch_num, costMean, costSymmetric, costSparsity, optmAll, Yinput, prediction, lambdaStep, softThr, transField = BM.build_model(trainPhi)


# 4. Model Training
print('-------------------------------------\nTraining Model...\n-------------------------------------\n')
TM.train_model(sess, saver, costMean, costSymmetric, costSparsity, optmAll, Yinput, prediction, trainLabel, valLabel, trainPhi, Xinput, Xoutput, Epoch_num, lambdaStep, softThr, missing_index, transField)
print('-------------------------------------\nTraining Accomplished.\n-------------------------------------\n')







