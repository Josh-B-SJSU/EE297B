%% Neural Network Generation
clc; clear; close all; % Start with a clean slate

%% Prepare Data
trainImageFolder = 'C:\';
trainMaskFolder = 'C:\';
valImageFolder = 'C:\';
valMaskFolder = 'C:\';

classes = ["background", "body", "panel"];
ids = { ... 
    % "Not Satellite"
    [
    0 0 0; ... % Black
    ]
    
    % "Body" 
    [
    0 255 0; ... % Green
    ]

    % "Panel" 
    [
    76 0 0; ... % Dark red
    ]
};

% create training data
imds_train = imageDatastore(trainImageFolder);
pxds_train = pixelLabelDatastore(trainMaskFolder,classes,ids);
trainingData = combine(imds_train,pxds_train);

% create val data
imds_val = imageDatastore(valImageFolder);
pxds_val = pixelLabelDatastore(valMaskFolder,classes,ids);
valData = combine(imds_val,pxds_val);

%% Define Neural Net
% setup deeplabv3+ with resnet
imageSize = [256 256 3];
numClasses = 3;
network = "resnet18";
net = deeplabv3plus(imageSize,numClasses,network);

% modify output layer to match
diceLossLayer = dicePixelClassificationLayer('Name', 'diceLoss');
lgraph = layerGraph(net);
lgraph = replaceLayer(lgraph, 'softmax-out', diceLossLayer);

options = trainingOptions('adam',...
    'InitialLearnRate',1e-3,...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'MaxEpochs', 100,...
    'MiniBatchSize', 32,...
    'ValidationData', valData,...
    'Plots','training-progress',...
    'Verbose', true, ...
    'Shuffle', 'every-epoch');

%% Train the model
trainedNet = trainNetwork(trainingData,lgraph,options);