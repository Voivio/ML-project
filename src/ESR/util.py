import numpy as np
import cv2
import os

class sample:
    def __init__(self, img, truth, box):
        self.img = img
        self.truth = truth
        # a nparray containing npts landmarks, each landmark is denoted by its x and y coordinate
        self.guess = np.zeros(truth.shape)
        self.box = box
        # a nparray of two coordinates, denoting the start and end point of a rectangle

def loadTrainData(data_path):
    number_files = len(os.listdir(data_path)) / 4 # because for a image we have 4 related files
    train_set = []
    for i in range(number_files):
        if i % 50 == 0:
            print('Samples loaded {} / {} ...'.format(i, number_files))

        img = cv2.imread(data_path + 'image_%04d.png'%(i+1))

        file = open('lfpw-test/image_%04d_original.ljson'%(i+1),'r')
        content = json.load(file)
        pts = content['groups'][0]['landmarks']
        for j in range(len(pts)):
            pts[i] = pts[i]['point']
            pts[i][0] = int(float(pts[i][0]))
            pts[i][1] = int(float(pts[i][1]))
            pts[i] = pts[i][::-1]

        file = open('lfpw-test/image_%04d_original.ljson'%(i+1),'r')
        content = json.load(file)
        rect = content['landmarks']["points"]
        rect[0][0] = int(float(rect[0][0]))
        rect[0][1] = int(float(rect[0][1]))
        rect[2][0] = int(float(rect[2][0]))
        rect[2][1] = int(float(rect[2][1]))
        rect = [rect[0][::-1], rect[2][::-1]]

        train_set.append(sample(img, np.array(pts), np.array(rect)))
    return train_set

def initialization(init_train_set, N_aug, stage = 'train'):
    number_samples = len(init_train_set)
    train_set = []
    # when training we use permuted truth as initial state
    if stage == 'train':
        for sample_index in len(init_train_set):
            random_index = np.random.permutation(number_samples)[:N_aug]
            for index in range(N_aug):
                train_set.append(init_train_set[sample_index])
                train_set[-1].guess = init_train_set[random_index[index]].truth

                # align the guess shape with the box
                train_set[-1].guess = alignShapeToBox(train_set[-1].guess, init_train_set[random_index[index]].box, train_set[-1].box);
        print('Initialization done. Number of augumented samples: {} x {} = {}'.format(number_samples, N_aug, number_samples*N_aug))
    else:
    # when testing, we take representive shape from train set
        pass

def alignShapeToBox(shape0, box0, box):
    npts = shape0.shape[0] # number of landmarks
    # shape = reshape(shape0, npts, 2);
    shape = np.zeros(shape0.shape)

    scale = box[1,0] / box0[1,0]
    # align the center of the shape to the center of the box
    box_c_x, boc_c_y = np.mean(box, 0)
    shape = shape0 - np.tile(np.mean(shape0, 0), (npts, 1))
    shape = shape .* scale
    shape = shape + np.tile([xc, yc], (npts, 1))

    return shape

def computeMeanShape(train_set):
    # compute in a iterative fashion:
    # 1) using truth shape(dataset.guess) of the first image as meanshape
    # 2) align all other truth shape to meanshape
    # 3) take average of all shapes as meanshape
    # 4) repeat 2)-3) until condition is met
    refshape = train_set[0].guess.reshape(1, -1)
    npts = refshape.size / 2
    # align all other shapes to this shape
    nshapes = len(train_set)
    alignedShapes = zeros(nshapes, npts*2)
    for i in range(nshapes)
        alignedShapes[i, :] = train_set[i].guess
    refshape = alignedShapes[1, :]

    iters = 0
    diff = float("inf")
    maxIters = 4
    while diff > 1e-2 && iters < maxIters:
        iters = iters + 1
        for i in range(nshapes):
            alignedShapes(i,:) = alignShape(alignedShapes(i,:), refshape)

        refshape_new = np.mean(alignedShapes, 0)
        diff = np.abs(np.max(refshape - refshape_new))
        refshape = refshape_new

    print('MeanShape finished in {} iterations.\n'.format(iters))
    return refshape.reshape(-1, 2)

def alignShape(s1, s0):
    npts = len(s1)/2
    s1 = s1.reshape(npts, 2)
    s0 = s0.reshape(npts, 2)
    [s, R, t] = estimateTransform(s1, s0)
    s1 = s * R * s1.T + tile(t, (1, npts))
    s1 = s1.T
    s1 = s1.reshape(1, npts*2)
    return s1

def estimateTransform(source_shape, target_shape):
    n, m = source_shape.shape

    mu_source = mean(source_shape, 0)
    mu_target = mean(target_shape, 0)

    d_source = source_shape - tile(mu_source, (n, 1))
    sig_source2 = np.sum(d_source*d_source)/n

    d_target = target_shape - repmat(mu_target, n, 1)
    sig_target2 = np.sum(d_target*d_target))/n

    sig_source_target = d_target.T.dot(d_source) / n

    det_sig_source_target = np.linalg.det(sig_p_target)
    S = np.eye(m)
    if det_sig_source_target < 0:
        S[n-1, m-1] = -1

    u, d, vh = np.linalg.svd(sig_source_target, full_matrices=True)

    R = u*d.dot(vh)
    s = np.trace(d*S)/sig_source2
    t = mu_target.T - s * R.dot(mu_p.T)
    return s, R, t

def normalizedShapeTargets
    nsamples = numel(trainset)
    Lfp = length(meanShape)
    Nfp = Lfp/2
    Mnorm = cell(nsamples, 1)
    Y = zeros(nsamples, Lfp)
    for i=1:nsamples
        [s, R, ~] = estimateTransform(reshape(trainset{i}.guess, Nfp, 2), ...
            reshape(meanShape, Nfp, 2))
        Mnorm{i}.M = s*R
        Mnorm{i}.invM = inv(Mnorm{i}.M)
        diff = trainset{i}.truth - trainset{i}.guess
        tdiff = Mnorm{i}.M * reshape(diff, Nfp, 2)'
        Y(i,:) = reshape(tdiff', 1, Lfp)

def learnStageRegressor(trainset, Y, Mnorm, opts)
    Lfp = length(trainset{1}.truth)
    Nfp = Lfp/2
    P = opts.params.P
    T = opts.params.T
    F = opts.params.F
    K = opts.params.K
    beta = opts.params.beta kappa = opts.params.kappa

    # generate local coordinates
    print('generate local coordinates...')
    localCoords = zeros(P, 3)  # fpidx, x, y
    for i in range(P):
        localCoords(i, 1) = randperm(Nfp, 1)
        localCoords(i, 2:3) = (rand(1, 2) - 0.5) * kappa

    # extract shape indexed pixels
    print('extract shape indexed pixels...')
    nsamples = numel(trainset)
    Mrho = zeros(nsamples, P)
    for i in range(nsamples):
        Minv = Mnorm{i}.invM

        dp = Minv * localCoords(:,2:3)'
        dp = dp'

        fpPos = reshape(trainset{i}.guess, Nfp, 2)
        pixPos = fpPos(ind2sub([Nfp 2],localCoords(:,1)), :) + dp
        [rows, cols] = size(trainset{i}.image)
        pixPos = round(pixPos)
        pixPos(:,1) = min(max(pixPos(:,1), 1), cols)
        pixPos(:,2) = min(max(pixPos(:,2), 1), rows)
        Mrho(i,:) = trainset{i}.image(sub2ind(size(trainset{i}.image), pixPos(:,2)', pixPos(:,1)'))
    # compute pixel-pixel covariance
    covRho = cov(Mrho)

    Mrho_centered = Mrho - repmat(mean(Mrho), size(Mrho, 1), 1)

    diagCovRho = diag(covRho)
    varRhoDRho = -2.0 * covRho + repmat(diagCovRho, 1, P) + repmat(diagCovRho', P, 1)
    inv_varRhoDRho = 1.0 ./ varRhoDRho

    # compute all ferns
    print('construct ferns...')
    ferns = cell(K,1)
    features = cell(K, 1)
    for k in range(K):
        print('Internal regressor {} / {}'.format(k, K))
        features{k} = correlationBasedFeatureSelection(Y, Mrho, Mrho_centered, inv_varRhoDRho, F)
        ferns{k} = trainFern(features{k}, Y, Mrho, beta)

        # update the normalized target
        Mdiff_rho = zeros(nsamples, F)
        for f=1:F
            Mdiff_rho(:,f) = features{k}{f}.rho_m - features{k}{f}.rho_n
        end
        updateMat = evaluateFern_batch(Mdiff_rho, ferns{k})
        fprintf('fern(%d)\tmax(Y) = %.6g, min(Y) = %.6g\n', k, max(max(Y)), min(min(Y)))
        Y = Y - updateMat

    regressor.localCoords = localCoords
    regressor.ferns = ferns
    regressor.features = features

    return regressor

def correlationBasedFeatureSelection(Y, Mrho, Mrho_centered, inv_varRhoDRho, F):
    [~, Lfp] = size(Y)
    Nfp = Lfp/2
    [n, P] = size(Mrho)
    features = cell(F, 1)

    for i in range(F):
        nu = randn(Lfp, 1)
        Yprob = Y * nu

        covYprob_rho = (sum(bsxfun(@times, Yprob-mean(Yprob), Mrho_centered)))/(n-1)
        covYprob_rho = covYprob_rho'
        #covYprob_rho = covVM(Yprob, Mrho_centered)

        #varYprob = var(Yprob)
        #inv_varYprob = 1.0 / sqrt(varYprob)

        covRhoMcovRho = repmat(covYprob_rho, 1, P) - repmat(covYprob_rho', P, 1)

        #corrYprob_rhoDrho = covRhoMcovRho .* (inv_varYprob * inv_varRhoDRho)
        corrYprob_rhoDrho = covRhoMcovRho .* inv_varRhoDRho

        #corrYprob_rhoDrho(logical(eye(size(corrYprob_rhoDrho)))) = -10000.0

        for j in range(P):
            corrYprob_rhoDrho(j, j) = -10000.0

        [maxCorr, maxLoc] = max(corrYprob_rhoDrho(:))
        [maxLoc_row, maxLoc_col] = ind2sub(size(corrYprob_rhoDrho), maxLoc)

        f.m = maxLoc_row
        f.n = maxLoc_col
        f.rho_m = Mrho(:,f.m)
        f.rho_n = Mrho(:,f.n)
        f.coor_rhoDiff = maxCorr
        features{i} = f

    return features

# def covVM(v, M_centered):
#     [n, ~] = size(M_centered)
#
#     mu_v = mean(v)
#     res = sum( bsxfun(@times, v-mu_v, M_centered) ) / (n-1)
#     res = res'
#     return res

# fern training
def trainFern(features, Y, Mrho, beta):
    F = numel(features)
    # compute thresholds for ferns
    thresholds = rand(F, 1)
    for f in range(F):
        fdiff = features{f}.rho_m - features{f}.rho_n
        maxval = max(fdiff)
        minval = min(fdiff)
        meanval = mean(fdiff)
        range = min(maxval-meanval, meanval-minval)
        thresholds(f) = (thresholds(f)-0.5)*0.2*range + meanval

    # partition the samples into 2^F bins
    bins = partitionSamples(Mrho, features, thresholds)

    # compute the outputs of each bin
    outputs = computeBinOutputs(bins, Y, beta)

    fern.thresholds = thresholds
    fern.outputs = outputs

    return fern

def partitionSamples(Mrho, features, thresholds):
    F = numel(features)
    bins = cell(2^F, 1)
    [nsamples, ~] = size(Mrho)
    diffvecs = zeros(nsamples, F)
    for i in range(F):
        diffvecs(:,i) = Mrho(:, features{i}.m) - Mrho(:, features{i}.n)

    for i in range(F):
        di = diffvecs(:,i)
        lset = find(di < thresholds(i))
        rset = setdiff(1:nsamples, lset)
        diffvecs(lset, i) = 0
        diffvecs(rset, i) = 1

    wvec = 2.^[0:F-1]'

    idxvec = diffvecs * wvec + 1

    for i=1:2^F
        bins{i} = find(idxvec==i)

    return bins

def computeBinOutputs(bins, Y, beta):
    [~, Lfp] = size(Y)
    nbins = numel(bins)
    outputs = zeros(nbins, Lfp)
    for i=1:nbins
        if isempty(bins{i})
            continue
        end

        outputs(i,:) = sum(Y(bins{i}, :))
        ni = length(bins{i})
        factor = 1.0 / ((1 + beta/ni)*ni)
        outputs(i,:) = outputs(i,:) * factor
        return outputs
