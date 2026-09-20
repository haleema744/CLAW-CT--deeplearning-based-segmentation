%% ========================================================================
%  CLAW SEGMENTATION WITH DEEPLABV3+ (ResNet50 Backbone, 2D Input)
%  ------------------------------------------------------------------------
%  This script performs leave-one-out cross-validation (LOOCV) for
%  semantic segmentation of claw CT slices into three classes:
%     0 = background, 1 = bone (red mask), 2 = keratin (green mask)
%
%  Pipeline overview:
%    1. Load CT volumes and RGB masks from disk.
%    2. Build 2D inputs (each slice is treated independently).
%    3. For each fold: split data into train / val / test (LOOCV).
%    4. Train DeepLabv3+ with ResNet50 backbone (ImageNet pretrained)
%       with validation monitoring.
%    5. Evaluate on the held-out claw and accumulate metrics.
%    6. Save per-fold models, confusion matrices, and summary results.
%
%  Requirements: Deep Learning Toolbox, Image Processing Toolbox,
%                Deep Learning Toolbox Model for ResNet-50 Network,
%                Computer Vision Toolbox (for deeplabv3plusLayers).
%% ========================================================================

clc;                % Clear the command window for a clean run
clear;              % Remove all variables from the workspace
close all;          % Close all open figures

%% ------------------ 1. PATHS -----------------------------
% Root folder that contains all claw subfolders (c1/, c1_1/, c2/, ...)
basePath = "C:\Users\s2278791\Music\CLAW_data";

% Total number of claws to attempt to load (skipped ones are ignored)
numClaws = 14;

% Pre-allocate cell arrays (one slot per claw) for memory efficiency
allClawVolumes = cell(numClaws, 1);   % Each cell: H x W x S CT volume
allClawMasks   = cell(numClaws, 1);   % Each cell: H x W x S label volume
allClawSlices  = cell(numClaws, 1);   % Each cell: number of loaded slices
allX = cell(numClaws, 1);             % Each cell: 2D input samples (single slices)
allY = cell(numClaws, 1);             % Each cell: matching label maps

% Track claws that successfully loaded (used later to filter empty cells)
loadedClaws = [];

% Human-readable names for each claw, e.g., {'c1','c2',...}
clawNames = cell(numClaws, 1);

%% ------------------ 2. LOAD ALL CLAWS -----------------------
% Print a banner to the console
disp('========================================');
disp('LOADING ALL CLAWS');
disp('========================================');

% Verify base path exists before attempting to read anything
if ~exist(basePath, 'dir')
    error('Base path does not exist: %s\nPlease check your folder path.', basePath);
else
    fprintf('Base path found: %s\n', basePath);
end

% List every subfolder inside basePath (used for debug/verification)
allFolders  = dir(basePath);
allFolders  = allFolders([allFolders.isdir]);       % Keep only directories
folderNames = {allFolders.name};                    % Extract names into a cell
fprintf('\nFolders found in base path:\n');
for i = 1:length(folderNames)
    % Skip "." and ".." special directory entries
    if ~strcmp(folderNames{i}, '.') && ~strcmp(folderNames{i}, '..')
        fprintf('  %s\n', folderNames{i});
    end
end
fprintf('\n');

% Main loop: load each claw one at a time
for clawIdx = 1:numClaws
    fprintf('\n========================================\n');
    fprintf('Loading Claw %d of %d...\n', clawIdx, numClaws);

    % Construct folder names following the project convention
    imgFolderName  = sprintf('c%d', clawIdx);              % e.g., 'c1', 'c2'
    maskFolderName = sprintf('c%d_%d', clawIdx, clawIdx);  % e.g., 'c1_1'

    % Build absolute paths to the image and mask folders
    imgFolder  = fullfile(basePath, imgFolderName);
    maskFolder = fullfile(basePath, maskFolderName);

    % Debug print the paths being checked
    fprintf('  Image folder: %s\n', imgFolder);
    fprintf('  Annotation folder: %s\n', maskFolder);

    % Skip claw if the image folder is missing
    if ~exist(imgFolder, 'dir')
        fprintf('  x Image folder NOT found: %s\n', imgFolder);
        continue;
    else
        fprintf('  + Image folder found\n');
    end

    % Skip claw if the mask folder is missing
    if ~exist(maskFolder, 'dir')
        fprintf('  x Annotation folder NOT found: %s\n', maskFolder);
        continue;
    else
        fprintf('  + Annotation folder found\n');
    end

    % List all JPG image files inside the image folder
    imgFiles = dir(fullfile(imgFolder, "*.jpg"));
    if isempty(imgFiles)
        fprintf('  x No JPG images found in: %s\n', imgFolder);
        continue;
    else
        fprintf('  + Found %d JPG images\n', length(imgFiles));
    end

    % List all JPG mask files inside the mask folder
    maskFiles = dir(fullfile(maskFolder, "*.jpg"));
    if isempty(maskFiles)
        fprintf('  x No JPG masks found in: %s\n', maskFolder);
        continue;
    else
        fprintf('  + Found %d JPG masks\n', length(maskFiles));
    end

    % Sort image files alphabetically so slice order is consistent
    [~, idx] = sort({imgFiles.name});
    imgFiles = imgFiles(idx);

    % Sort mask files alphabetically to match image order
    [~, idx] = sort({maskFiles.name});
    maskFiles = maskFiles(idx);

    % Handle mismatch between number of images and masks
    if length(imgFiles) ~= length(maskFiles)
        fprintf('  ! Warning: Number of images (%d) and masks (%d) do not match\n', ...
            length(imgFiles), length(maskFiles));
        numSlices = min(length(imgFiles), length(maskFiles));   % Take the minimum
        fprintf('  Using first %d slices\n', numSlices);
    else
        numSlices = length(imgFiles);   % Counts match: use all
    end

    fprintf('  Total slices to load: %d\n', numSlices);

    % Skip claws that have fewer than 1 slice
    if numSlices < 1
        fprintf('  x Not enough slices (need at least 1, found %d)\n', numSlices);
        continue;
    end

    % Pre-allocate volume containers (unknown dimensions until first slice)
    volume  = [];
    maskVol = [];

    % Counters for tracking progress and errors
    loadedCount = 0;
    errorCount  = 0;

    fprintf('  Loading slices...\n');
    for i = 1:numSlices
        try
            % ---- Load CT image slice and convert to double [0,1] ----
            imgPath = fullfile(imgFolder, imgFiles(i).name);
            I = im2double(imread(imgPath));

            % If the image is RGB, convert to grayscale for storage
            if size(I, 3) == 3
                I = rgb2gray(I);
            end

            % ---- Load corresponding RGB mask slice ----
            maskPath = fullfile(maskFolder, maskFiles(i).name);
            M = imread(maskPath);

            % Resize mask if its spatial size does not match image size
            if size(M, 1) ~= size(I, 1) || size(M, 2) ~= size(I, 2)
                M = imresize(M, [size(I, 1), size(I, 2)], 'nearest');
                if errorCount < 5
                    fprintf('  ! Slice %d: Resized mask to match image dimensions\n', i);
                end
            end

            % ---- Convert RGB mask into a single-channel label map ----
            if size(M, 3) == 3
                % Split color channels
                R = M(:,:,1);
                G = M(:,:,2);
                B = M(:,:,3);

                % Initialize label map to background (0)
                label = zeros(size(R), 'uint8');

                % Class 0: background (near black pixels)
                label(R < 50 & G < 50 & B < 50) = 0;

                % Class 1: bone (dominant red pixels)
                label(R > 150 & G < 80 & B < 80) = 1;

                % Class 2: keratin (dominant green pixels)
                label(G > 150 & R < 80 & B < 80) = 2;
            else
                % Grayscale masks are assumed to already encode 0/1/2
                label = uint8(M);
            end

            % Store slice into the 3D volume buffers
            volume(:,:,i)  = I;
            maskVol(:,:,i) = label;
            loadedCount = loadedCount + 1;

            % Print progress every 100 slices
            if mod(i, 100) == 0
                fprintf('    Loaded %d/%d slices\n', i, numSlices);
            end

        catch ME
            % Log errors but continue loading remaining slices
            errorCount = errorCount + 1;
            if errorCount <= 5
                fprintf('  ! Error loading slice %d: %s\n', i, ME.message);
            elseif errorCount == 6
                fprintf('  ! Additional errors suppressed...\n');
            end
            continue;
        end
    end

    % Skip claw if no slice was loaded
    if loadedCount == 0
        fprintf('  x No slices loaded for claw %d\n', clawIdx);
        continue;
    end

    % Skip claw if no slices loaded
    if loadedCount < 1
        fprintf('  x Only %d slices loaded (need at least 1)\n', loadedCount);
        continue;
    end

    % Trim trailing empty slices if some failed mid-way
    if loadedCount < numSlices
        volume  = volume(:,:,1:loadedCount);
        maskVol = maskVol(:,:,1:loadedCount);
        fprintf('  ! Trimmed to %d valid slices\n', loadedCount);
    end

    % Store loaded data into cell arrays
    allClawVolumes{clawIdx} = volume;
    allClawMasks{clawIdx}   = maskVol;
    allClawSlices{clawIdx}  = loadedCount;

    % Mark this claw as successfully loaded
    loadedClaws = [loadedClaws, clawIdx];
    clawNames{clawIdx} = sprintf('c%d', clawIdx);

    fprintf('  + Claw %d successfully loaded with %d slices\n', clawIdx, loadedCount);
end

% List of claws that were successfully loaded
validIndices  = loadedClaws;
numValidClaws = length(validIndices);

% Abort if no claw loaded at all
if numValidClaws == 0
    fprintf('\n========================================\n');
    fprintf('ERROR: No claws were loaded successfully.\n');
    fprintf('Please check the following:\n');
    fprintf('1. Base path: %s\n', basePath);
    fprintf('2. Folder structure should be:\n');
    fprintf('   %s\n', basePath);
    fprintf('   |-- c1\\ (contains JPG files directly)\n');
    fprintf('   |-- c1_1\\ (contains JPG files directly)\n');
    fprintf('   |-- c2\\\n');
    fprintf('   |-- c2_2\\\n');
    fprintf('   +-- ...\n');
    fprintf('========================================\n');
    error('No claws loaded. Please fix your folder structure.');
end

% Keep only successfully loaded claws
allClawVolumes = allClawVolumes(validIndices);
allClawMasks   = allClawMasks(validIndices);
allClawSlices  = allClawSlices(validIndices);
clawNames      = clawNames(validIndices);

% Summarize loading result
fprintf('\n========================================');
fprintf('\nSuccessfully loaded %d out of %d claws', numValidClaws, numClaws);
fprintf('\nLoaded claws: %s', mat2str(validIndices));
fprintf('\n========================================\n');

%% ------------------ 3. PREPARE 2D DATA FOR EACH CLAW -----------------------
disp(' ');
disp('========================================');
disp('PREPARING 2D DATA FOR ALL CLAWS');
disp('========================================');

% DeepLabv3+ input spatial resolution (ResNet50-friendly)
networkSize = 224;

% Class names and label IDs (order matters for categorical conversion)
classNames  = {'background','bone','keratin'};
labelIDs    = [0 1 2];

% Loop through each valid claw to build 2D samples (one slice = one sample)
for clawIdx = 1:numValidClaws
    fprintf('\nProcessing Claw %d of %d (%s)...\n', clawIdx, numValidClaws, clawNames{clawIdx});

    % Fetch the volume and mask for this claw
    volume    = allClawVolumes{clawIdx};
    maskVol   = allClawMasks{clawIdx};
    numSlices = size(volume, 3);

    fprintf('  Total slices: %d\n', numSlices);

    % Skip claw if there are no slices
    if numSlices < 1
        fprintf('  Warning: Claw %s has no slices, skipping...\n', clawNames{clawIdx});
        continue;
    end

    % Pre-allocate sample containers (one sample per slice)
    X = cell(numSlices, 1);
    Y = cell(numSlices, 1);
    count = 1;

    fprintf('  Creating 2D samples...\n');

    % Loop over each slice independently
    for i = 1:numSlices
        try
            % ---- Extract the single slice ----
            slice = volume(:,:,i);

            % ---- Min-max normalize to [0,1] ----
            minVal = min(slice(:));
            maxVal = max(slice(:));
            if maxVal > minVal
                slice = (slice - minVal) / (maxVal - minVal);
            else
                slice = zeros(size(slice));   % Flat slice -> zeros
            end

            % ---- Resize to network input size ----
            slice = imresize(slice, [networkSize, networkSize]);

            % ---- Replicate grayscale to 3 channels for ResNet50 ----
            sliceRGB = cat(3, slice, slice, slice);

            % ---- Store the 3-channel image ----
            X{count} = sliceRGB;

            % ---- Label = the same slice's mask ----
            labelMatrix = maskVol(:,:,i);

            % Resize label map with nearest-neighbor (preserve class IDs)
            labelMatrix = imresize(labelMatrix, [networkSize, networkSize], 'nearest');

            % Convert to categorical with explicit class ordering
            Y{count} = categorical(labelMatrix, [0 1 2], classNames);

            count = count + 1;

        catch ME
            fprintf('  Error creating sample %d: %s\n', i, ME.message);
            continue;
        end
    end

    % Trim unused pre-allocated slots
    X = X(1:count-1);
    Y = Y(1:count-1);

    % Store samples for this claw
    allX{clawIdx} = X;
    allY{clawIdx} = Y;

    fprintf('  OK Claw %s: %d samples prepared (from %d slices)\n', ...
        clawNames{clawIdx}, length(X), numSlices);
end

% Drop claws that produced zero samples
validSamples = find(~cellfun(@isempty, allX));
if length(validSamples) < numValidClaws
    fprintf('\nWarning: Some claws did not produce any samples. Keeping only valid ones.\n');
    allX           = allX(validSamples);
    allY           = allY(validSamples);
    allClawVolumes = allClawVolumes(validSamples);
    allClawMasks   = allClawMasks(validSamples);
    clawNames      = clawNames(validSamples);
    numValidClaws  = length(clawNames);
end

%% ------------------ 4. DISPLAY DATA SUMMARY -----------------------
fprintf('\n========================================');
fprintf('\nDATA SUMMARY');
fprintf('\n========================================\n');

totalSamples = 0;
for clawIdx = 1:numValidClaws
    numSamples = length(allX{clawIdx});
    totalSamples = totalSamples + numSamples;
    fprintf('Claw %s: %d samples\n', clawNames{clawIdx}, numSamples);
end
fprintf('Total samples across all claws: %d\n', totalSamples);

% Sanity check: ensure at least one sample exists
if totalSamples == 0
    error('No samples were created. Please check your data.');
end

%% ------------------ 5. LEAVE-ONE-OUT WITH VALIDATION SET -----------------------
fprintf('\n========================================');
fprintf('\nSTARTING LOOCV WITH VALIDATION SET (DEEPLABV3+ RESNET50)');
fprintf('\n========================================\n');
fprintf('\nConfiguration: %d Training, 2 Validation, 1 Test per fold', numValidClaws - 3);
fprintf('\nTotal folds: %d\n', numValidClaws);

% Class configuration (re-declared for clarity inside the LOOCV block)
classNames = {'background','bone','keratin'};
labelIDs   = [0 1 2];
numClasses = 3;

% Pre-allocate result containers per fold
allTestAccuracies       = zeros(numValidClaws, 1);      % Per-fold test accuracy
allValidationAccuracies = cell(numValidClaws, 1);       % Per-fold validation curves
allTrainingAccuracies   = cell(numValidClaws, 1);       % Per-fold training curves
allConfusionMatrices    = cell(numValidClaws, 1);       % Per-fold confusion matrices
allPredictions          = cell(numValidClaws, 1);       % Predicted map (middle slice)
allGroundTruth          = cell(numValidClaws, 1);       % GT map (middle slice)
allBestEpochs           = zeros(numValidClaws, 1);      % Best epoch per fold
allFoldInfo             = cell(numValidClaws, 1);       % Train/val/test split info

% ============ Main LOOCV loop ============
for testClawIdx = 1:numValidClaws
    fprintf('\n========================================');
    fprintf('\nFOLD %d of %d', testClawIdx, numValidClaws);
    fprintf('\nTesting on Claw: %s', clawNames{testClawIdx});
    fprintf('\n========================================\n');

    % Remaining claws (all except current test claw)
    remainingClaws = setdiff(1:numValidClaws, testClawIdx);
    numRemaining   = length(remainingClaws);

    % Fix RNG seed per fold for reproducibility
    rng(testClawIdx);

    % Shuffle remaining claws to randomize train/val split
    shuffleIdx     = randperm(numRemaining);
    remainingClaws = remainingClaws(shuffleIdx);

    % Split ~85% training / ~15% validation
    numTrainClaws = max(1, round(numRemaining * 0.85));
    numValClaws   = numRemaining - numTrainClaws;

    % Assign train and validation claw indices
    trainClaws = remainingClaws(1:numTrainClaws);
    valClaws   = remainingClaws(numTrainClaws+1:end);

    % Store fold split information for later inspection
    allFoldInfo{testClawIdx} = struct(...
        'testClaw',  clawNames{testClawIdx}, ...
        'trainClaws', {clawNames(trainClaws)}, ...
        'valClaws',   {clawNames(valClaws)});

    % Build a printable string of training claw names
    trainClawNames = '';
    for i = 1:length(trainClaws)
        if i == 1
            trainClawNames = clawNames{trainClaws(i)};
        else
            trainClawNames = [trainClawNames, ', ', clawNames{trainClaws(i)}]; %#ok<AGROW>
        end
    end

    % Build a printable string of validation claw names
    valClawNames = '';
    for i = 1:length(valClaws)
        if i == 1
            valClawNames = clawNames{valClaws(i)};
        else
            valClawNames = [valClawNames, ', ', clawNames{valClaws(i)}]; %#ok<AGROW>
        end
    end

    % Report the fold split
    fprintf('Training claws: %s\n', trainClawNames);
    fprintf('Validation claws: %s\n', valClawNames);
    fprintf('Test claw: %s\n', clawNames{testClawIdx});

    % Initialize aggregated sample cells
    trainX = {}; trainY = {};
    valX   = {}; valY   = {};

    % Aggregate training samples across all training claws
    for idx = 1:length(trainClaws)
        clawIdx = trainClaws(idx);
        trainX  = [trainX; allX{clawIdx}]; %#ok<AGROW>
        trainY  = [trainY; allY{clawIdx}]; %#ok<AGROW>
    end

    % Aggregate validation samples across all validation claws
    for idx = 1:length(valClaws)
        clawIdx = valClaws(idx);
        valX    = [valX; allX{clawIdx}];   %#ok<AGROW>
        valY    = [valY; allY{clawIdx}];   %#ok<AGROW>
    end

    % Report sample counts
    fprintf('Training samples: %d\n',   length(trainX));
    fprintf('Validation samples: %d\n', length(valX));
    fprintf('Test samples: %d\n',       length(allX{testClawIdx}));

    % -------- Create temp directory for datastore files --------
    tempDir = fullfile(pwd, 'temp_segmentation_data_fold');
    if exist(tempDir, 'dir')
        rmdir(tempDir, 's');            % Remove stale files from previous run
    end
    mkdir(tempDir);

    % Create subfolders for training images/labels
    trainImageDir = fullfile(tempDir, 'train_images');
    trainLabelDir = fullfile(tempDir, 'train_labels');
    mkdir(trainImageDir);
    mkdir(trainLabelDir);

    % Save training images and labels to .mat files on disk
    fprintf('Saving training data...');
    for i = 1:length(trainX)
        img = trainX{i};
        imgFile = fullfile(trainImageDir, sprintf('img_%04d.mat', i));
        save(imgFile, 'img');

        label = trainY{i};
        labelFile = fullfile(trainLabelDir, sprintf('label_%04d.mat', i));
        save(labelFile, 'label');
    end
    fprintf(' Done.\n');

    % Create subfolders for validation images/labels
    valImageDir = fullfile(tempDir, 'val_images');
    valLabelDir = fullfile(tempDir, 'val_labels');
    mkdir(valImageDir);
    mkdir(valLabelDir);

    % Save validation images and labels to .mat files on disk
    fprintf('Saving validation data...');
    for i = 1:length(valX)
        img = valX{i};
        imgFile = fullfile(valImageDir, sprintf('img_%04d.mat', i));
        save(imgFile, 'img');

        label = valY{i};
        labelFile = fullfile(valLabelDir, sprintf('label_%04d.mat', i));
        save(labelFile, 'label');
    end
    fprintf(' Done.\n');

    % Build imageDatastore for training images using a custom reader
    trainImds = imageDatastore(trainImageDir, 'FileExtensions', '.mat', ...
        'ReadFcn', @(f) loadImageFromFile(f));

    % Build pixelLabelDatastore for training labels
    trainPxds = pixelLabelDatastore(trainLabelDir, classNames, labelIDs, ...
        'FileExtensions', '.mat', ...
        'ReadFcn', @(f) loadLabelFromFile(f));

    % Combine image and label datastores for training
    dsTrain = pixelLabelImageDatastore(trainImds, trainPxds);

    % Build imageDatastore for validation images
    valImds = imageDatastore(valImageDir, 'FileExtensions', '.mat', ...
        'ReadFcn', @(f) loadImageFromFile(f));

    % Build pixelLabelDatastore for validation labels
    valPxds = pixelLabelDatastore(valLabelDir, classNames, labelIDs, ...
        'FileExtensions', '.mat', ...
        'ReadFcn', @(f) loadLabelFromFile(f));

    % Combine validation image and label datastores
    dsVal = pixelLabelImageDatastore(valImds, valPxds);

    % ================================================================
    %  BUILD DEEPLABV3+ WITH RESNET50 BACKBONE (2D SINGLE-SLICE INPUT)
    %  ----------------------------------------------------------------
    %  MATLAB built-in function: deeplabv3plusLayers
    %  - Encoder: ResNet50 (ImageNet pretrained)
    %  - Decoder: atrous spatial pyramid pooling (ASPP) + upsampling
    %  - Input shape: [224 224 3] (grayscale replicated to 3 channels)
    %  - Output: per-pixel class scores + softmax + pixelClassification
    % ================================================================

    % DeepLabv3+ input image size (3-channel due to ResNet50)
    imageSize = [224, 224, 3];

    try
        % Construct DeepLabv3+ with ResNet50 backbone
        lgraph = deeplabv3plusLayers(imageSize, numClasses, 'resnet50');
        fprintf('Using DeepLabv3+ with ResNet50 backbone\n');
    catch ME
        % If deeplabv3plusLayers is unavailable, report the error clearly
        fprintf('Error creating DeepLabv3+: %s\n', ME.message);
        fprintf('Ensure Computer Vision Toolbox and ResNet50 support package are installed.\n');
        rethrow(ME);
    end

    % Optional: display network summary in the command window
    try
        analyzeNetwork(lgraph);
    catch
        % analyzeNetwork can be slow; skip silently if it fails
    end

    % -------- Training Options --------
    options = trainingOptions("adam", ...                  % Adam optimizer
        "MaxEpochs", 30, ...                               % Maximum epochs
        "MiniBatchSize", 8, ...                            % Batch size (DeepLabv3+ is memory heavy)
        "Shuffle", "every-epoch", ...                      % Shuffle each epoch
        "Plots", "training-progress", ...                  % Show live plot
        "Verbose", true, ...                               % Print progress
        "ValidationData", dsVal, ...                       % Validation datastore
        "ValidationFrequency", 30, ...                     % Validate every 30 iters
        "ValidationPatience", 10, ...                      % Early stopping patience
        "InitialLearnRate", 1e-4, ...                      % Starting LR
        "L2Regularization", 0.0001, ...                    % Weight decay
        "LearnRateSchedule", "piecewise", ...              % Piecewise LR schedule
        "LearnRateDropFactor", 0.1, ...                    % LR drop factor
        "LearnRateDropPeriod", 10, ...                     % Drop every 10 epochs
        "GradientThreshold", 1.0, ...                      % Clip gradients
        "ExecutionEnvironment", "auto", ...                % Use GPU if available
        "OutputNetwork", "best-validation-loss");          % Return best net

    % -------- Train Network --------
    fprintf('Training Fold %d (DeepLabv3+ ResNet50)...\n', testClawIdx);
    [net, info] = trainNetwork(dsTrain, lgraph, options);

    % Extract training and validation accuracy curves if available
    if isfield(info, 'TrainingLoss') && isfield(info, 'ValidationLoss')
        allTrainingAccuracies{testClawIdx}   = info.TrainingAccuracy;
        allValidationAccuracies{testClawIdx} = info.ValidationAccuracy;

        % Identify epoch with best validation accuracy
        [maxValAcc, bestEpoch] = max(info.ValidationAccuracy);
        allBestEpochs(testClawIdx) = bestEpoch;
        fprintf('OK Best model at epoch %d with validation accuracy: %.2f%%\n', ...
            bestEpoch, maxValAcc);
    end

    % Save trained model for this fold
    save(['deeplabv3plus_model_fold_' num2str(testClawIdx) '.mat'], 'net');
    fprintf('OK DeepLabv3+ model saved as: deeplabv3plus_model_fold_%d.mat\n', testClawIdx);

    % -------- Test on held-out claw --------
    testX = allX{testClawIdx};
    testY = allY{testClawIdx};

    fprintf('Testing on %d samples...\n', length(testX));

    % Pre-allocate prediction and ground truth containers
    predictions  = cell(length(testX), 1);
    groundTruth  = cell(length(testX), 1);

    % Run semantic segmentation on every test sample
    for i = 1:length(testX)
        pred = semanticseg(testX{i}, net);   % Predict label map
        predictions{i} = pred;
        groundTruth{i} = testY{i};
    end

    % Aggregate all pixel predictions and ground-truth labels
    allPred = [];
    allGT   = [];

    for i = 1:length(predictions)
        pred = predictions{i};
        gt   = groundTruth{i};

        % Convert categorical maps to numeric label IDs
        predNumeric = double(pred);
        gtNumeric   = double(gt);

        % Resize back to original slice size for visualization
        originalSize = size(allClawVolumes{testClawIdx}(:,:,1));
        predResized  = imresize(predNumeric, originalSize, 'nearest');
        gtResized    = imresize(gtNumeric,   originalSize, 'nearest');

        % Keep middle slice for visualization
        if i == round(length(predictions)/2)
            allPredictions{testClawIdx}  = predResized;
            allGroundTruth{testClawIdx}  = gtResized;
        end

        % Append flattened pixel labels for confusion matrix
        allPred = [allPred; predNumeric(:)]; %#ok<AGROW>
        allGT   = [allGT;   gtNumeric(:)];   %#ok<AGROW>
    end

    % Compute confusion matrix for this fold
    cm = confusionmat(allGT, allPred);
    allConfusionMatrices{testClawIdx} = cm;

    % Overall pixel accuracy for this fold
    accuracy = sum(diag(cm)) / sum(cm(:));
    allTestAccuracies(testClawIdx) = accuracy;

    fprintf('\nOK Fold %d DeepLabv3+ Test Accuracy: %.2f%%\n', testClawIdx, accuracy * 100);

    % Compute per-class precision, recall, F1
    fprintf('Per-class metrics for Fold %d:\n', testClawIdx);
    for c = 1:numClasses
        tp = cm(c,c);                        % True positives for class c
        fp = sum(cm(:,c)) - tp;              % False positives for class c
        fn = sum(cm(c,:)) - tp;              % False negatives for class c
        precision = tp / (tp + fp + eps);    % Precision
        recall    = tp / (tp + fn + eps);    % Recall
        f1        = 2 * precision * recall / (precision + recall + eps); % F1
        fprintf('  %s - Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%\n', ...
            classNames{c}, precision*100, recall*100, f1*100);
    end

    % Cleanup: remove temporary datastore directory
    if exist(tempDir, 'dir')
        rmdir(tempDir, 's');
    end

    % Close training plots to avoid clutter across folds
    close all;
end

%% ------------------ 6. OVERALL RESULTS -----------------------
fprintf('\n========================================');
fprintf('\nLEAVE-ONE-OUT CROSS-VALIDATION RESULTS (DEEPLABV3+ RESNET50)');
fprintf('\n========================================\n');

% Aggregate statistics across folds
meanAcc = mean(allTestAccuracies);
stdAcc  = std(allTestAccuracies);
minAcc  = min(allTestAccuracies);
maxAcc  = max(allTestAccuracies);

fprintf('\nOverall Test Accuracy:\n');
fprintf('  Mean: %.2f%%\n', meanAcc * 100);
fprintf('  Std: +/- %.2f%%\n', stdAcc * 100);
fprintf('  Min: %.2f%%\n', minAcc * 100);
fprintf('  Max: %.2f%%\n', maxAcc * 100);

% Best epoch statistics across folds
fprintf('\nBest Epoch Statistics:\n');
fprintf('  Mean: %.1f\n', mean(allBestEpochs));
fprintf('  Std: +/- %.1f\n', std(allBestEpochs));
fprintf('  Min: %d\n', min(allBestEpochs));
fprintf('  Max: %d\n', max(allBestEpochs));

% Compute average confusion matrix across folds
avgCM = zeros(numClasses, numClasses);
for i = 1:numValidClaws
    if ~isempty(allConfusionMatrices{i})
        avgCM = avgCM + allConfusionMatrices{i};
    end
end
avgCM = avgCM / numValidClaws;

fprintf('\nAverage Confusion Matrix:\n');
disp(avgCM);

% Compute average per-class metrics from the averaged confusion matrix
fprintf('\nAverage Per-class Metrics:\n');
for c = 1:numClasses
    tp = avgCM(c,c);
    fp = sum(avgCM(:,c)) - tp;
    fn = sum(avgCM(c,:)) - tp;
    precision = tp / (tp + fp + eps);
    recall    = tp / (tp + fn + eps);
    f1        = 2 * precision * recall / (precision + recall + eps);
    fprintf('  %s - Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%\n', ...
        classNames{c}, precision*100, recall*100, f1*100);
end

%% ------------------ 7. VISUALIZATION -----------------------
% Figure 1: Grid of predicted segmentation maps per claw
figure('Position', [100, 100, 1800, 800]);

for i = 1:min(numValidClaws, 16)
    subplot(4, 4, i);

    if ~isempty(allPredictions{i})
        % Convert numeric label map to colored RGB for display
        predRGB = label2rgb(allPredictions{i}, [0 0 0; 1 0 0; 0 1 0], 'k');
        imshow(predRGB);
        title(sprintf('%s - Acc: %.1f%%', clawNames{i}, allTestAccuracies(i)*100));
    else
        % Placeholder text when no prediction is available
        text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
        title(clawNames{i});
    end
end

sgtitle('LOOCV DeepLabv3+ (ResNet50) Results for All Claws');

% Figure 2: Bar plot of test accuracy per claw
figure('Position', [100, 100, 1000, 600]);
bar(allTestAccuracies * 100);
xlabel('Claw Number');
ylabel('Test Accuracy (%)');
title('LOOCV DeepLabv3+ (ResNet50) Test Accuracy per Claw');
ylim([0 100]);
grid on;

% Annotate each bar with its numeric value
for i = 1:numValidClaws
    text(i, allTestAccuracies(i)*100 + 1, sprintf('%.1f', allTestAccuracies(i)*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end

% Use claw names as x-axis tick labels
set(gca, 'XTickLabel', clawNames);

% Figure 3: Boxplot of test accuracy distribution
figure('Position', [100, 100, 600, 400]);
boxplot(allTestAccuracies * 100);
ylabel('Test Accuracy (%)');
title('LOOCV DeepLabv3+ (ResNet50) Test Accuracy Distribution');
grid on;

%% ------------------ 8. SAVE RESULTS -----------------------
% Bundle all results into a single struct for reproducibility
results.foldAccuracies        = allTestAccuracies;
results.meanAccuracy          = meanAcc;
results.stdAccuracy           = stdAcc;
results.confusionMatrices     = allConfusionMatrices;
results.averageConfusionMatrix= avgCM;
results.classNames            = classNames;
results.bestEpochs            = allBestEpochs;
results.trainingAccuracies    = allTrainingAccuracies;
results.validationAccuracies  = allValidationAccuracies;
results.clawNames             = clawNames;
results.foldInfo              = allFoldInfo;
results.networkType           = 'DeepLabv3+ (ResNet50 backbone, 2D single-slice)';

% Save results to disk
save('LOOCV_deeplabv3plus_results_with_validation.mat', 'results');
fprintf('\nOK Results saved to LOOCV_deeplabv3plus_results_with_validation.mat\n');

fprintf('\n========================================');
fprintf('\nDEEPLABV3+ (RESNET50) PROCESS COMPLETED SUCCESSFULLY!');
fprintf('\n========================================\n');

%% =========================================================
%  HELPER FUNCTIONS
%  NOTE: Local functions must be placed at the end of the script.
%% =========================================================

% ---- loadImageFromFile ----
% Custom reader for imageDatastore that loads a .mat file and returns
% the variable 'img' as the image.
