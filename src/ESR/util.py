import numpy as np
import cv2
import os
from scipy.io import loadmat
import json
import pdb


# class sample:
#     def __init__(self, img, truth, box):
#         self.image = img
#         self.truth = truth
#         # a nparray containing npts landmarks, each landmark is denoted by its x and y coordinate
#         self.guess = np.zeros(truth.shape)
#         self.box = box
#         # a nparray of two coordinates, denoting the start and end point of a rectangle
#
#
# class regressor:
#     def __init__(self, localCoords, ferns, features):
#         self.localCoords = localCoords
#         self.ferns = ferns
#         self.features = features
#
#
# class feature:
#     def __init__(self, m, n, rho_m, rho_n, coor_rhoDiff):
#         self.m = m
#         self.n = n
#         self.rho_m = rho_m
#         self.rho_n = rho_n
#         self.coor_rhoDiff = coor_rhoDiff
#
#
# class fern:
#     def __init__(self, thresholds, outputs):
#         self.thresholds = thresholds
#         self.outputs = outputs
#
#
# def loadTrainData(data_path):
#     number_files = len(os.listdir(data_path)) / 4  # because for a image we have 4 related files
#     train_set = []
#     for i in range(number_files):
#         if i % 50 == 0:
#             print('Samples loaded {} / {} ...'.format(i, number_files))
#
#         img = cv2.imread(data_path + 'image_%04d.png' % (i + 1), cv2.IMREAD_GRAYSCALE)
#
#         file = open('lfpw-test/image_%04d_original.ljson' % (i + 1), 'r')
#         content = json.load(file)
#         pts = content['groups'][0]['landmarks']
#         for j in range(len(pts)):
#             pts[i] = pts[i]['point']
#             pts[i][0] = int(float(pts[i][0]))
#             pts[i][1] = int(float(pts[i][1]))
#             pts[i] = pts[i][::-1]
#
#         file = open('lfpw-test/image_%04d_original.ljson' % (i + 1), 'r')
#         content = json.load(file)
#         rect = content['landmarks']["points"]
#         rect[0][0] = int(float(rect[0][0]))
#         rect[0][1] = int(float(rect[0][1]))
#         rect[2][0] = int(float(rect[2][0]))
#         rect[2][1] = int(float(rect[2][1]))
#         rect = [rect[0][::-1], rect[2][::-1]]
#
#         train_set.append(sample(img, np.array(pts), np.array(rect)))
#     return train_set
#
#
# def getDistPupils(shape):
#     npts = shape.shape[0]
#     if npts == 29:
#         dist_pupils = np.linalg.norm(shape[7 - 1, :] - shape[16 - 1, :])
#     elif npts == 68:
#         left_eye_4 = [38 - 1, 39 - 1, 41 - 1, 42 - 1]
#         right_eye_4 = [44 - 1, 45 - 1, 47 - 1, 48 - 1]
#         left_center = np.mean(shape(left_eye_4,:), 0)
#         right_center = np.mean(shape(right_eye_4,:), 0)
#         dist_pupils = np.linalg.norm(left_center - right_center)
#
#     return dist_pupils
#
#
# def initialization(init_train_set, N_aug, stage='train'):
#     number_samples = len(init_train_set)
#     train_set = []
#     # when training we use permuted truth as initial state
#     if stage == 'train':
#         for sample_index in len(init_train_set):
#             random_index = np.random.permutation(number_samples)[:N_aug]
#             for index in range(N_aug):
#                 train_set.append(init_train_set[sample_index])
#                 train_set[-1].guess = init_train_set[random_index[index]].truth
#
#                 # align the guess shape with the box
#                 train_set[-1].guess = alignShapeToBox(train_set[-1].guess, init_train_set[random_index[index]].box,
#                                                       train_set[-1].box)
#         print('Initialization done. Number of augumented samples: {} x {} = {}'.format(number_samples, N_aug,
#                                                                                        number_samples * N_aug))
#     else:
#         # when testing, we take representive shape from train set
#         pass
#
#
def alignShapeToBox(shape0, box0, box):
    pdb.set_trace()
    npts = shape0.shape[0]  # number of landmarks
    # shape = reshape(shape0, npts, 2)
    shape = np.zeros(shape0.shape)

    scale = box[0] / box0[0]
    # align the center of the shape to the center of the box
    box_c_x, boc_c_y = np.mean(box, 0)
    shape = shape0 - np.tile(np.mean(shape0, 0), (npts, 1))
    shape = shape * scale
    shape = shape + np.tile([xc, yc], (npts, 1))

    return shape


#
#
# def estimateTransform(source_shape, target_shape):
#     n, m = source_shape.shape
#
#     mu_source = np.mean(source_shape, 0)
#     mu_target = np.mean(target_shape, 0)
#
#     d_source = source_shape - tile(mu_source, (n, 1))
#     sig_source2 = np.sum(d_source * d_source) / n
#
#     d_target = target_shape - repmat(mu_target, n, 1)
#     sig_target2 = np.sum(d_target * d_target)) / n
#
#     sig_source_target = d_target.T.dot(d_source) / n
#
#     det_sig_source_target = np.linalg.det(sig_p_target)
#     S = np.eye(m)
#     if det_sig_source_target < 0:
#         S[n - 1, m - 1] = -1
#
#     u, d, vh = np.linalg.svd(sig_source_target, full_matrices=True)
#
#     R = u * d.dot(vh)
#     s = np.trace(d * S) / sig_source2
#     t = mu_target.T - s * R.dot(mu_p.T)
#     return s, R, t
#
#
# def computeMeanShape(train_set):
#     # compute in a iterative fashion:
#     # 1) using truth shape(dataset.guess) of the first image as meanshape
#     # 2) align all other truth shape to meanshape
#     # 3) take average of all shapes as meanshape
#     # 4) repeat 2)-3) until condition is met
#     refshape = train_set[0].guess.reshape(1, -1)
#     npts = refshape.size / 2
#     # align all other shapes to this shape
#     nshapes = len(train_set)
#     alignedShapes = zeros(nshapes, npts * 2)
#     for i in range(nshapes)
#         alignedShapes[i, :] = train_set[i].guess
#     refshape = alignedShapes[1, :]
#
#     iters = 0
#     diff = float("inf")
#     maxIters = 4
#     while diff > 1e-2 & & iters < maxIters:
#         iters = iters + 1
#         for i in range(nshapes):
#             alignedShapes(i,:) = alignShape(alignedShapes(i,:), refshape)
#
#             refshape_new = np.mean(alignedShapes, 0)
#             diff = np.abs(np.max(refshape - refshape_new))
#             refshape = refshape_new
#
#         print('MeanShape finished in {} iterations.\n'.format(iters))
#         return refshape.reshape(-1, 2)
#
#     def alignShape(s1, s0):
#         npts = len(s1) / 2
#         s1 = s1.reshape(npts, 2)
#         s0 = s0.reshape(npts, 2)
#         [s, R, t] = estimateTransform(s1, s0)
#         s1 = s * R * s1.T + tile(t, (1, npts))
#         s1 = s1.T
#         s1 = s1.reshape(1, npts * 2)
#         return s1
#
#     def normalizedShapeTargets(train_set, mean_shape):
#         nsamples = len(train_set)
#         npts = mean_shape.shape[0]
#         M_norm = []  # M_norm contains the similarity transform matrix for each sample
#         # Mnorm = cell(nsamples, 1)
#         Y = np.zeros(nsamples, npts)
#         for i in range(nsamples):
#             [s, R, ~] = estimateTransform(trainset[i].guess, mean_shape)
#             M_norm.append(s * R)
#             # Mnorm{i}.invM = inv(Mnorm{i}.M)
#             diff = trainset[i].truth - trainset[i].guess
#             tdiff = M_norm[i].dot(diff.T)
#             Y(i,:) = tdiff.T.reshape(1, -1)
#         return Y, M_norm
#
#     def learnStageRegressor(train_set, Y, M_norm, params):
#         npts = trainset[0].truth.shape[0]
#         P = params['P']
#         T = params['T']
#         F = params['F']
#         K = params['K']
#         beta = params['beta']
#         kappa = params['kappa']
#
#         # generate local coordinates
#         print('Generating local coordinates...')
#         localCoords = np.zeros(P, 3)  # fpidx, x, y
#         for i in range(P):
#             localCoords[i, 0] = np.randint(0, npts)  # randomly choose a landmark
#             localCoords[i, 1:] = (np.random.uniform(size=(1, 2)) - 0.5) * kappa  # fluctuate around landmark
#
#         # extract shape indexed pixels
#         print('Extracting shape indexed pixels...')
#         nsamples = len(train_set)
#         M_rho = np.zeros(nsamples, P)
#         for i in range(nsamples):
#             M_norm_inv = np.linalg.inv(M_norm[i])
#
#             dp = M_norm_inv.dot(localCoords[:, 1:].T).T
#
#             # fpPos = reshape(train_set[i].guess, Nfp, 2)
#             # pixPos = fpPos(ind2sub([Nfp 2],localCoords(:,1)), :) + dp
#
#             pixPos = train_set[i].guess(localCoords[:, 0],:) + dp
#             rows, cols = trainset[i].image.shape
#             pixPos = np.round(pixPos)
#             pixPos(:, 0) = np.minimum(np.maximum(pixPos[:, 0], 0), cols - 1)
#             pixPos(:, 1) = np.minimum(np.maximum(pixPos[:, 1], 0), rows - 1)
#             # in case pixel position out of range
#             M_rho[i, :] = trainset[i].image[pixPos(:, 2).T, pixPos(:, 1).T]
#             # compute pixel-pixel covariance
#             cov_Rho = np.cov(M_rho, rowvar=False)
#
#             M_rho_centered = M_rho - tile(mean(M_rho, 0), (M_rho.shape[0], 1))
#
#             diagCovRho = np.diag(cov_Rho)
#             varRhoDRho = -2.0 * covRho + repmat(diagCovRho.T, 1, P) + repmat(diagCovRho, P, 1)
#             inv_varRhoDRho = 1.0 / varRhoDRho  # element-wise inverse
#
#             # compute all ferns
#             print('Constructing ferns...')
#             ferns = []
#             features = []
#             for k in range(K):
#                 features.append(correlationBasedFeatureSelection(Y, M_rho, M_rho_centered, inv_varRhoDRho, F))
#                 ferns.append(trainFern(features[-1], Y, M_rho, beta))
#
#                 # update the normalized target
#                 M_diff_rho = np.zeros(nsamples, F)
#                 for f in range(F):
#                     M_diff_rho[:, f] = features[k, f].rho_m - features[k, f].rho_n
#
#                 updateMat = evaluateFern_batch(M_diff_rho, ferns[k])
#                 print('fern %d/%d\tmax(Y) = %.6g, min(Y) = %.6g' % (k, K, np.max(Y), np.min(Y)))
#                 Y = Y - updateMat
#
#             regressor = regressor(localCoords, ferns, features)
#
#             return regressor
#
#         def correlationBasedFeatureSelection(Y, M_rho, M_rho_centered, inv_varRhoDRho, F):
#             Lfp = Y.shape[1]
#             Nfp = Lfp / 2
#             n, P = M_rho.shape
#             features = []
#
#             for i in range(F):
#                 nu = np.random.randn(Lfp, 1)
#                 Yprob = Y.dot(nu)
#
#                 covYprob_rho = (sum(Yprob - mean(Yprob) * M_rho_centered), 0) / (n - 1)  # R^{1xP}
#                 covRhoMcovRho = tile(covYprob_rho.T, (1, P)) - tile(covYprob_rho, (P, 1))
#                 corrYprob_rhoDrho = covRhoMcovRho * np.sqrt(inv_varRhoDRho)
#
#                 # corrYprob_rhoDrho(logical(eye(size(corrYprob_rhoDrho)))) = -10000.0
#
#                 for j in range(P):
#                     corrYprob_rhoDrho[j, j] = -10000.0
#
#                 maxCorr = max(corrYprob_rhoDrho)
#                 maxLoc_row, maxLoc_col = np.unravel_index(np.argmax(corrYprob_rhoDrho, axis=None),
#                                                           corrYprob_rhoDrho.shape)
#
#                 features.append(feature(maxLoc_row, maxLoc_col, Mrho[:, f.m], Mrho[:, f.n], maxCorr))
#
#             return features
#
#         # def covVM(v, M_centered):
#         #     [n, ~] = size(M_centered)
#         #
#         #     mu_v = mean(v)
#         #     res = sum( bsxfun(@times, v-mu_v, M_centered) ) / (n-1)
#         #     res = res'
#         #     return res
#
#         # fern training
#         def trainFern(features, Y, Mrho, beta):
#             F = len(features)
#             # compute thresholds for ferns
#             thresholds = np.random.uniform(size=(F, 1))
#             for f in range(F):
#                 fdiff = features[f].rho_m - features[f].rho_n
#                 maxval = max(fdiff)
#                 minval = min(fdiff)
#                 meanval = np.mean(fdiff)
#                 range = min(maxval - meanval, meanval - minval)
#                 thresholds[f] = (thresholds[f] - 0.5) * 0.2 * range + meanval
#
#             # partition the samples into 2^F bins
#             bins = partitionSamples(Mrho, features, thresholds)
#
#             # compute the outputs of each bin
#             outputs = computeBinOutputs(bins, Y, beta)
#
#             fern = fern(thresholds, outputs)
#             # fern.thresholds = thresholds
#             # fern.outputs = outputs
#             return fern
#
#         def partitionSamples(Mrho, features, thresholds):
#             F = len(features)
#             # bins = cell(2^F, 1)
#             binss = []
#             nsamples = Mrho.shape[0]
#             diffvecs = np.zeros(nsamples, F)
#             for i in range(F):
#                 diffvecs[:, i] = Mrho[:, features[i].m] - Mrho[:, features[i].n]
#
#             for i in range(F):
#                 di = diffvecs[:, i]
#                 lset = np.where(di < thresholds[i])
#                 rset = np.setdiff1d(array(range(nsamples)), lset)
#                 diffvecs[lset, i] = 0
#                 diffvecs[rset, i] = 1
#
#             wvec = np.array(range(F))
#             wvec = 2 ** wvec[:, np.newaxis]
#
#             idxvec = diffvecs.dot(wvec)
#
#             for i in range(2 ** F):
#                 bins.append(np.where(idxvec == i))
#
#             return bins
#
#         def computeBinOutputs(bins, Y, beta):
#             Lfp = Y.shape[1]
#             nbins = len(bins)
#             outputs = np.zeros(nbins, Lfp)
#             for i in range(nbins):
#                 if bins[i].size == 0:  # empty bin
#                     continue
#
#                 outputs[i, :] = sum(Y[bins[i], :])
#                 ni = len(bins[i])
#                 factor = 1.0 / ((1 + beta / ni) * ni)
#                 outputs[i, :] = outputs[i, :] * factor
#             return outputs
#
#         def evaluateFern_batch(diffvecs, fern):
#             F = len(fern.thresholds)
#             nsamples = diffvecs.shape[0]
#             for i in range(F):
#                 di = diffvecs[:, i]
#                 lset = np.where(di < thresholds[i])
#                 rset = np.setdiff1d(array(range(nsamples)), lset)
#                 diffvecs[lset, i] = 0
#                 diffvecs[rset, i] = 1
#
#             wvec = np.array(range(F))
#             wvec = 2 ** wvec[:, np.newaxis]
#
#             idxvec = diffvecs.dot(wvec)
#
#             output = fern.outputs[idxvec, :]
#             return output
#
#         def updateGuessShapes(trainset, Mnorm, regressor):
#             nsamples = len(trainset)
#             Nfp = trainset[0].truth.shape[0]
#             maxError = 0
#             F = len(regressor.ferns[0].thresholds)
#             K = len(regressor.ferns)
#             rho_diff = np.zeros(nsamples, F)
#             Mds = np.zeros(nsamples, Nfp * 2)
#             for k in range(K):
#                 for f in range(F):
#                     rho_diff[:, f] = regressor.features[k, f].rho_m - regressor.features[k, f].rho_n
#
#                 Mds = Mds + evaluateFern_batch(rho_diff, regressor.ferns[k])
#
#             for i in range(nsamples):
#                 ds = Mds[i, :]
#                 ds = ds.reshape(Nfp, 2).T
#                 ds = np.linalg.inv(Mnorm[i]).dot(ds)
#                 ds = ds.T
#                 trainset[i].guess = trainset[i].guess + ds
#                 error = (trainset[i].truth - trainset[i].guess).reshape(-1, 1)
#                 maxError = max(maxError, np.linalg.norm(error))
#
#             print('Maxerror : {}'.format(maxError))
#             return trainset


# class FernClassifier:
#
#
# class Stage:
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def from_json_dict(d):
#         localCoords = np.array(['localCoords'], dtype=np.float)

class Model:
    def __init__(self, init_boxes, init_shapes, meanshape, stages):
        self.init_boxes = init_boxes
        self.init_shapes = init_shapes
        self.meanshape = meanshape
        self.stages = stages

    @staticmethod
    def load_model(dirname):
        init_boxes = loadmat(os.path.join(dirname, 'model_init_boxes.mat'))['big_mat']  # (2584, 136)
        init_shapes = loadmat(os.path.join(dirname, 'model_init_shapes.mat'))['big_mat']  # (4722, 4)
        meanshape = loadmat(os.path.join(dirname, 'model_meanshape.mat'))['meanshape']  # (1, 136)
        stages = []
        for i in range(1, 11):
            with open(os.path.join(dirname, 'model_stages_{:d}.json'.format(i))) as f:
                stages.append(json.load(f))
        model = Model(init_boxes, init_shapes, meanshape, stages)
        return model


if __name__ == '__main__':
    model = Model.load_model('../pipeline/models')
