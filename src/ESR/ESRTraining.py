import util
import pickle as pkl

# Load training dataset
data_path = '../../data/lfpw/lfpw-train'
init_train_set = util.loadTrainData(data_path)

# Initialization
train_set = util.initialization(init_train_set, N_aug, stage = 'train')

# compute mean shape0
meanShape = util.computeMeanShape(train_set)

for t in range(T):
    # external stage
    print('Stage {} / {}'.format(t, T))

    # compute normalized shape targets
    [Y, Mnorm] = util.normalizedShapeTargets(trainset, meanshape)

    # learn stage regressor, including
    # 1) feature generation 2) feature selection 3) learning internal regressors
    ex_regressors{t} = util.learnStageRegressor(trainset, Y, Mnorm, opts)

    # update guess shapes for all samples
    trainset = util.updateGuessShapes(trainset, Mnorm, stages{t})

# save model
pkl.dump(ex_regressors)
