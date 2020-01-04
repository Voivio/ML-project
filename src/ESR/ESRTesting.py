import util
import cv2
import h5py
# import numpy as np
# filepath = 'model.mat'
# arrays = {}
# f = h5py.File(filepath)
# for k, v in f.items():
#     arrays[k] = np.array(v)

def applyModel(img, model):
    # Load a face detector and an image
    cascade_filepath = '../opencv-4.2.0/data/haarcascades'
    detector = cv2.CascadeClassifier([cascade_filepath, '/', 'haarcascade_frontalface_alt.xml'])

    # Preprocess
    h, w, channels = img.shape
    if channels > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gr = cv2.equalizeHist(img)
    I = img

    # Detect bounding box
    boxes = detector.detectMultiScale(gr, scaleFactor = 1.3, minNeighbors = 2, minSize = (30, 30))

    try:
        box = boxes[0]
    except IndexError as e:
        boxes = np.array([])
        points = np.array([])
        succeeded = False
        return

    # Scale it properly, actually no sclae
    # take only on box
    # for bidx = 1:numel(boxes)
    bsize = box.width
    box = [box.x, box.y, box.x + bsize, box.y + box.height]
    # boxSize = boxes{1}(3)

    # scale the image
    # box = (boxes{bidx}) * sfactor
    nh, nw = I.shape

    # cut out a smaller region
    # bsize = box[2]
    # bcenter = [box(1) + 0.5 * bsize, box(2) + 0.5 * bsize]
    bcenter = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    # enlarge this region
    cutsize = 4.0 * bsize / 2
    nbx_tl = max(1, round(bcenter[0] - cutsize))
    nby_tl = max(1, round(bcenter[1] - cutsize))
    nbx_br = min(nw, round(bcenter[0] + cutsize))
    nby_br = min(nh, round(bcenter[1] + cutsize))

    # cut out image
    newimg = I[nby_tl:nby_br, nbx_tl:nbx_br]
    # get the new bounding box
    newbox = box
    newbox[0] = newbox[0] - nbx_tl
    newbox[1] = newbox[1] - nby_tl
    newbox[2] = newbox[2] + nbx_br
    newbox[3] = newbox[3] + nby_br


    ntrials = 5
    idx = np.random.permutation(len(model.init_shapes))[:ntrials]
    Lfp = 68*2
    Nfp = Lfp/2
    results = np.zeros(ntrials, Lfp)
    T = numel(model.stages)
    F = numel(model.stages{1}.ferns{1}.thresholds)
    K = numel(model.stages{1}.ferns)
    meanshape = model.meanshape

    for i in range(ntrials):
        # get an initial guess
        guess = model.init_shapes[idx[i]]

        # align the guess to the bounding box
        guess = alignShapeToBox(guess, model.init_boxes[idx[i]], newbox)

        # find the points using the model
        for t in range(T):
            [s, R, ~] = estimateTransform(guess.reshape(Nfp, 2), meanshape.reshape(Nfp, 2))
            M = s*R
            lc = model.stages[t].localCoords
            P = lc.shape[0]
            dp = np.linalg.pinv(M).dot(lc(:,1:2).T)
            dp = dp.T
            # fpPos = guess.reshape(Nfp, 2)
            pixPos = fpPos(lc[:,0], :) + dp
            # pixPos = fpPos[ind2sub([Nfp 2],lc(:,1)), :] + dp
            rows, cols = newimg.shape
            pixPos = np.round(pixPos)
            pixPos(:,0) = np.minimum(np.maximum(pixPos[:,0], 0), cols-1)
            pixPos(:,1) = np.minimum(np.maximum(pixPos[:,1], 0), rows-1)
            pix = newimg[pixPos(:,2).T, pixPos(:,1).T]
            # pix = newimg(sub2ind(size(newimg), pixPos(:,2)', pixPos(:,1)'))

            ds = 0
            for k in range(K):
                rho = np.zeros(F, 1)
                for f in range(F):
                    m = model.stages[t].features{k}{f}.m
                    n = model.stages[t].features{k}{f}.n
                    rho[f] = pix[m] - pix[n]
                ds = ds + evaluateFern(rho, model.stages{t}.ferns{k})

            ds = ds.reshape(Nfp, 2).T
            ds = np.linalg.pinv(M).dot(ds)
            ds = ds.T.reshape(1, Lfp)
            guess = guess + ds
        results[i,:] = guess

    # # pick 5 best results
    # nearestNeighbors = knnsearch(results, mean(results), 'K', 25)
    points = np.median(results, 0))

    # restore the correct positions
    points = points.reshape(Nfp, 2)
    temp = np.array([nbx_tl, nby_tl])
    temp = temp[:,np.newaxis]
    points = points + tile(temp, (Nfp, 1))

    succeeded = True

return box, points, succeeded
