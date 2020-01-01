function [boxes, points, succeeded] = applyModel(img, model)

%% Load a face detector and an image
cascade_filepath = 'C:\Users\Peihong\Desktop\Code\Libraries\opencv\sources\data\haarcascades';
detector = cv.CascadeClassifier([cascade_filepath, '\', 'haarcascade_frontalface_alt.xml']);

% Preprocess
[h, w, channels] = size(img);
if channels > 1
    img = rgb2gray(img);
end
gr = cv.equalizeHist(img);
I0 = img;

% Detect bounding box
boxes = detector.detect(gr, 'ScaleFactor',  1.3, ...
    'MinNeighbors', 2, ...
    'MinSize',      [30, 30]);

if isempty(boxes)
    boxes = [];
    points = [];
    succeeded = false;
    return;
end

points = cell(numel(boxes), 1);

% Scale it properly
for bidx = 1:numel(boxes)
    boxSize = boxes{1}(3);
    
    % scale the image
    sfactor = model.window_size / boxSize;
    if sfactor == 0.0
        sfactor = 1.0;
    end
    
    I = imresize(I0, sfactor);    
    box = (boxes{bidx}) * sfactor;
    [nh, nw] = size(I);
    
    % cut out a smaller region
    bsize = box(3);
    bcenter = [box(1) + 0.5 * bsize, box(2) + 0.5 * bsize];
    
    % enlarge this region
    cutsize = 4.0 * bsize / 2;
    nbx_tl = max(1, round(bcenter(1) - cutsize)); nby_tl = max(1, round(bcenter(2) - cutsize));
    nbx_br = min(nw, round(bcenter(1) + cutsize)); nby_br = min(nh, round(bcenter(2) + cutsize));

    % cut out image
    newimg = I(nby_tl:nby_br, nbx_tl:nbx_br);
    % get the new bounding box
    newbox = box;
    newbox(1) = newbox(1) - nbx_tl; newbox(2) = newbox(2) - nby_tl;
    
    ntrials = 75;
    idx = randperm(numel(model.init_shapes), ntrials);
    Lfp = 136; Nfp = Lfp/2;
    results = zeros(ntrials, Lfp);
    T = numel(model.stages); F = numel(model.stages{1}.ferns{1}.thresholds);
    K = numel(model.stages{1}.ferns);
    meanshape = model.meanshape;
    for i=1:ntrials
        % get an initial guess
        guess = cell2mat(model.init_shapes(idx(i)));
        
        % align the guess to the bounding box
        guess = alignShapeToBox(guess, model.init_boxes{idx(i)}, newbox);
        
        % find the points using the model
        for t=1:T
            [s, R, ~] = estimateTransform(reshape(guess, Nfp, 2), reshape(meanshape, Nfp, 2));
            M = s*R;
            lc = model.stages{t}.localCoords;
            [P,~] = size(lc);
            dp = M \ lc(:,2:3)';
            dp = dp';
            fpPos = reshape(guess, Nfp, 2);
            pixPos = fpPos(ind2sub([Nfp 2],lc(:,1)), :) + dp;
            [rows, cols] = size(newimg);
            pixPos = round(pixPos);
            pixPos(:,1) = min(max(pixPos(:,1), 1), cols);
            pixPos(:,2) = min(max(pixPos(:,2), 1), rows);
            pix = newimg(sub2ind(size(newimg), pixPos(:,2)', pixPos(:,1)'));

            ds = 0;
            for k=1:K
                rho = zeros(F, 1);
                for f=1:F
                    m = model.stages{t}.features{k}{f}.m;
                    n = model.stages{t}.features{k}{f}.n;
                    rho(f) = pix(m) - pix(n);
                end
                ds = ds + evaluateFern(rho, model.stages{t}.ferns{k});
            end
            
            ds = reshape(ds, Nfp, 2)';
            ds = M\ds;
            ds = reshape(ds', 1, Lfp);
            guess = guess + ds;
        end
        results(i,:) = guess;
    end
    
    % pick 5 best results
    nearestNeighbors = knnsearch(results, mean(results), 'K', 25);
    points{bidx} = median(results(nearestNeighbors, :));
    
    % restore the correct positions
    points{bidx} = reshape(points{bidx}, Nfp, 2);
    points{bidx} = points{bidx} + repmat([nbx_tl, nby_tl], Nfp, 1);    
    points{bidx} = points{bidx} / sfactor;
end
succeeded = true;
end