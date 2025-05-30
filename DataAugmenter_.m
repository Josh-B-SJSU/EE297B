%% Data Augmenter
clc; clear; close all; % Start with a clean slate

%% Define the Filepaths
inputImageFolder = 'C:\';
inputMaskFolder = 'C:\';
outputTrainImageFolder = 'C:\';
outputTrainMaskFolder = 'C:\';
outputTestImageFolder = 'C:\';
outputTestMaskFolder = 'C:\';

%% Load Images and Masks
% Create Datastores
imds = imageDatastore(inputImageFolder);
pxds = imageDatastore(inputMaskFolder);

% Determine Ratio of Train to Test
numImages = numel(imds.Files);
splitRatio = 0.8; % 80% train, 20% test
numTrain = round(splitRatio * numImages);

% Shuffle and Split the Data
idx = randperm(numImages);
trainIdx = idx(1:numTrain);
testIdx = idx(numTrain+1:end);

% Create Subsets for Training and Test
trainImds = subset(imds, trainIdx);
trainPxds = subset(pxds, trainIdx);
testImds = subset(imds, testIdx);
testPxds = subset(pxds, testIdx);

%% Define Augmentation Options
augmenter = imageDataAugmenter( ...
    'RandRotation', [-90, 90], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXTranslation', [-64, 64], ...
    'RandYTranslation', [-64, 64], ...
    'RandScale', [0.25 5] ...
);

%% Augment & Save Training Data
numAugmentations = 15; % Increase Dataset Size by 15x

for i = 1:numel(trainIdx)
    I = readimage(trainImds, i);
    M = readimage(trainPxds, i);

    % Save original image and mask
    imwrite(I, fullfile(outputTrainImageFolder, sprintf('image_%d.png', i)));
    imwrite(M, fullfile(outputTrainMaskFolder, sprintf('mask_%d.png', i)));

    for j = 1:numAugmentations
        % Apply identical augmentation to image and mask
        tform = randomAffine2d( ...
            'Rotation', [-90, 90], ...
            'XReflection', true, ...
            'XTranslation', [-64, 64], ...
            'YTranslation', [-64, 64], ...
            'Scale',[0.25 4]);
        rout = affineOutputView(size(I), tform);
        
        % Nearest neighbor to keep labels intact
        augI = imwarp(I, tform, 'OutputView', rout);
        augM = imwarp(M, tform, 'OutputView', rout);

        % Save augmented image and mask
        imwrite(augI, fullfile(outputTrainImageFolder, sprintf('image_%d_aug%d.png', i, j)));
        imwrite(augM, fullfile(outputTrainMaskFolder, sprintf('mask_%d_aug%d.png', i, j)));
    end
end

%% Save Test Data Without Augmentation
disp('Saving test data...');
for i = 1:numel(testIdx)
    I = readimage(testImds, i);
    M = readimage(testPxds, i);

    imwrite(I, fullfile(outputTestImageFolder, sprintf('image_%d.png', i)));
    imwrite(M, fullfile(outputTestMaskFolder, sprintf('mask_%d.png', i)));
end