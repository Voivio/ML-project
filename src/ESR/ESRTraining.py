import util
import pickle as pkl

params = {
    'P' : 400,
    'T' : 10,
    'F' : 5,
    'K' : 500,
    'N_aug' : 20,
    'beta' : 1000,
    'kappa' : 0
    }

# Load training dataset
data_path = '../../data/lfpw/lfpw-train'
init_train_set = util.loadTrainData(data_path)

# Initialization
train_set = util.initialization(init_train_set, params['N_aug'], stage = 'train')

# compute mean shape0
mean_shape = util.computeMeanShape(train_set)
params.kappa = 0.3 * getDistPupils(mean_shape) # kappa set to 0.3 * pulils dist in mean_hsape

stages = []
for t in range(T):
    # external stage
    print('Stage {} / {}'.format(t, T))

    # compute normalized shape targets
    [Y, M_norm] = util.normalizedShapeTargets(train_set, mean_shape)

    # learn stage regressor, including
    # 1) feature generation 2) feature selection 3) learning internal regressors
    stages.append(util.learnStageRegressor(train_set, Y, M_norm, params))

    # update guess shapes for all samples
    trainset = util.updateGuessShapes(trainset, Mnorm, stages[t])


    for k in range(len(stages[t].features)):
        for f in range(len((stages{t}.features[k]))):
            stages[t].features[k, f].rho_m = []
            stages[t].features[k, f].rho_n = []

# save model
model.window_size = opts.params.window_size;
pkl.dump(model(meanshape, init_shapes, init_boxes, stages), )
