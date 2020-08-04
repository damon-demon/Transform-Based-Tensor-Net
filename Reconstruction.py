#       Video Synthesis via Transform-Based Tensor Neural Network
#                             Yimeng Zhang
#                               8/4/2020
#                         yz3397@columbia.edu


import LoadData as LD
import BuildModel as BM
import ReconstructionImage as RI
import DefineParam as DP
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 1. Input: Parameters
pixel_w, pixel_h, batchSize, nPhase, nTrainData, nValData, learningRate, nEpoch, nOfModel, ncpkt, trainFile, valFile, testFile, saveDir, modelDir = DP.get_param()


# 2. Data Loading
print('-------------------------------------\nLoading Data...\n-------------------------------------\n')
testLabel = LD.load_test_data(mat73=True)
testLabel /= 255
testPhi = np.ones([batchSize, 240, 320, 3])
missing_index = [1]
print(missing_index)

for index_x in missing_index:
    testPhi[:, :, :, index_x] = 0

testPhi = testPhi.astype('float32')


# 3. Model Building
print('-------------------------------------\nBuilding and Restoring Model...\n-------------------------------------\n')
sess, saver, Xinput, Xoutput, Yinput, Epoch_num, prediction, transField = BM.build_model(testPhi, restore=True)


# 4. Image reconstruction
print('-------------------------------------\nReconstructing Image...\n-------------------------------------\n')
RI.reconstruct_image(sess, Yinput, Epoch_num, prediction, transField, Xinput, Xoutput, testLabel, testPhi, missing_index)
print('-------------------------------------\nReconstructing Accomplished.\n-------------------------------------\n')





