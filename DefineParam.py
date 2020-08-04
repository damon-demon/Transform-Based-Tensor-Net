#       Video Synthesis via Transform-Based Tensor Neural Network
#                             Yimeng Zhang
#                               8/4/2020
#                         yz3397@columbia.edu

# Parameter Definitions
def get_param():    
    pixel_w = 320
    pixel_h = 240
    batchSize = 1
    nPhase = 5
    nTrainData = 2808
    nValData = 1396
    learningRate = 0.0001
    nEpoch = 300
    nOfModel = 3
    ncpkt = 290

    trainFile = './trainData/trainUCF.mat'
    valFile = './valData/valUCF.mat'
    testFile = './testData/testRGB_R.mat'
    saveDir = './recImg/recImgFinal'
    modelDir = './Model_Int'

    return pixel_w, pixel_h, batchSize, nPhase,nTrainData, nValData, learningRate, nEpoch, nOfModel, ncpkt, trainFile, valFile, testFile, saveDir, modelDir
