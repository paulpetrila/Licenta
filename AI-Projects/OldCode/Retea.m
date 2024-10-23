%% Setup, cleanup

clear all
close all
clc

%% Download dataset

%old way, from the flower generator
%url = "http://download.tensorflow.org/example_images/flower_photos.tgz";
%downloadFolder = '/Users/paul/Coding/facultate/ai-projects/GeneratingFlowers/flori';
%filename = fullfile(downloadFolder,"flower_dataset.tgz");

% imageFolder = fullfile(downloadFolder,"flower_photos");
% if ~exist(imageFolder,"dir")
%     disp("Downloading Flowers data set (218 MB)...")
%     websave(filename,url);
%     untar(filename,downloadFolder)
% end

imageFolder = fullfile('/Users/paul-cosminpetrila/Coding/Facultate/Licenta/archive (1)/natural_images/')



%% Dataset
datasetFolder = fullfile(imageFolder);
imds = imageDatastore(datasetFolder,IncludeSubfolders=true,LabelSource="foldernames");

augmenter = imageDataAugmenter(RandXReflection=true);
augimds = augmentedImageDatastore([64 64],imds,DataAugmentation=augmenter);

classes = categories(imds.Labels);
numClasses = numel(classes)
%% Arhitectura generator
numLatentInputs = 100;
embeddingDimension = 50;
numFilters = 64;

filterSize = 5;
projectionSize = [4 4 1024];

layersGenerator = [
    featureInputLayer(numLatentInputs)
    fullyConnectedLayer(prod(projectionSize))
    functionLayer(@(X) feature2image(X,projectionSize),Formattable=true)
    concatenationLayer(3,2,Name="cat");
    transposedConv2dLayer(filterSize,4*numFilters)
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(filterSize,2*numFilters,Stride=2,Cropping="same")
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(filterSize,numFilters,Stride=2,Cropping="same")
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(filterSize,3,Stride=2,Cropping="same")
    tanhLayer];

lgraphGenerator = layerGraph(layersGenerator);

layers = [
    featureInputLayer(1)
    embeddingLayer(embeddingDimension,numClasses)
    fullyConnectedLayer(prod(projectionSize(1:2)))
    functionLayer(@(X) feature2image(X,[projectionSize(1:2) 1]),Formattable=true,Name="emb_reshape")];

lgraphGenerator = addLayers(lgraphGenerator,layers);
lgraphGenerator = connectLayers(lgraphGenerator,"emb_reshape","cat/in2");
netG = dlnetwork(lgraphGenerator)


analyzeNetwork(netG)

%% Sectiune discriminator

dropoutProb = 0.75;
numFilters = 64;
scale = 0.2;

inputSize = [64 64 3];
filterSize = 5;

layersDiscriminator = [
    imageInputLayer(inputSize,Normalization="none")
    dropoutLayer(dropoutProb)
    concatenationLayer(3,2,Name="cat")
    convolution2dLayer(filterSize,numFilters,Stride=2,Padding="same")
    leakyReluLayer(scale)
    convolution2dLayer(filterSize,2*numFilters,Stride=2,Padding="same")
    batchNormalizationLayer
    leakyReluLayer(scale)
    convolution2dLayer(filterSize,4*numFilters,Stride=2,Padding="same")
    batchNormalizationLayer
    leakyReluLayer(scale)
    convolution2dLayer(filterSize,8*numFilters,Stride=2,Padding="same")
    batchNormalizationLayer
    leakyReluLayer(scale)
    convolution2dLayer(4,1)];

lgraphDiscriminator = layerGraph(layersDiscriminator);

layers = [
    featureInputLayer(1)
    embeddingLayer(embeddingDimension,numClasses)
    fullyConnectedLayer(prod(inputSize(1:2)))
    functionLayer(@(X) feature2image(X,[inputSize(1:2) 1]),Formattable=true,Name="emb_reshape")];

lgraphDiscriminator = addLayers(lgraphDiscriminator,layers);
lgraphDiscriminator = connectLayers(lgraphDiscriminator,"emb_reshape","cat/in2");

netD = dlnetwork(lgraphDiscriminator)

analyzeNetwork(netD)

%% sectiune de antrenare

% setup for the loop 

numEpochs = 300;
miniBatchSize = 64;
learnRate = 0.0002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;
validationFrequency = 100;
flipFactor = 0.5;

augimds.MiniBatchSize = miniBatchSize;
executionEnvironment = "auto";

mbq = minibatchqueue(augimds, ...
    MiniBatchSize=miniBatchSize, ...
    PartialMiniBatch="discard", ...
    MiniBatchFcn=@preprocessData, ...
    MiniBatchFormat=["SSCB" "BC"], ...
    OutputEnvironment=executionEnvironment);    
velocityD = [];
trailingAvgG = [];
trailingAvgSqG = [];
trailingAvgD = [];
trailingAvgSqD = [];
numValidationImagesPerClass = 5;
ZValidation = randn(numLatentInputs,numValidationImagesPerClass*numClasses,"single");

TValidation = single(repmat(1:numClasses,[1 numValidationImagesPerClass]));

ZValidation = dlarray(ZValidation,"CB");
TValidation = dlarray(TValidation,"CB");
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    ZValidation = gpuArray(ZValidation);
    TValidation = gpuArray(TValidation);
end

numObservationsTrain = numel(imds.Files);
numIterationsPerEpoch = floor(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor( ...
    Metrics=["GeneratorScore","DiscriminatorScore"], ...
    Info=["Epoch","Iteration"], ...
    XLabel="Iteration");

groupSubPlot(monitor,Score=["GeneratorScore","DiscriminatorScore"])

% training loop

epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Reset and shuffle data.
    reset(mbq)
    shuffle(mbq);

    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;

        % Read mini-batch of data.
        [X,T] = next(mbq);

        % Generate latent inputs for the generator network. Convert to
        % dlarray and specify the dimension labels "CB" (channel, batch).
        % If training on a GPU, then convert latent inputs to gpuArray.
        Z = randn(numLatentInputs,miniBatchSize,"single");
        Z = dlarray(Z,"CB");
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            Z = gpuArray(Z);
        end

        % Evaluate the gradients of the loss with respect to the learnable
        % parameters, the generator state, and the network scores using
        % dlfeval and the modelLoss function.
        [~,~,gradientsG,gradientsD,stateG,scoreG,scoreD] = ...
            dlfeval(@modelLoss,netG,netD,X,T,Z,flipFactor);
        netG.State = stateG;

        % Update the discriminator network parameters.
        [netD,trailingAvgD,trailingAvgSqD] = adamupdate(netD, gradientsD, ...
            trailingAvgD, trailingAvgSqD, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [netG,trailingAvgG,trailingAvgSqG] = ...
            adamupdate(netG, gradientsG, ...
            trailingAvgG, trailingAvgSqG, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Every validationFrequency iterations, display batch of generated images using the
        % held-out generator input.
        if mod(iteration,validationFrequency) == 0 || iteration == 1
            
            % Generate images using the held-out generator input.
            XGeneratedValidation = predict(netG,ZValidation,TValidation);
            
            % Tile and rescale the images in the range [0 1].
            I = imtile(extractdata(XGeneratedValidation), ...
                GridSize=[numValidationImagesPerClass numClasses]);
            I = rescale(I);
            
            % Display the images.
            image(I)
            xticklabels([]);
            yticklabels([]);
            title("Generated Images");
        end

        % Update the training progress monitor.
        recordMetrics(monitor,iteration, ...
            GeneratorScore=scoreG, ...
            DiscriminatorScore=scoreD);

        updateInfo(monitor,Epoch=epoch,Iteration=iteration);
        monitor.Progress = 100*iteration/numIterations;
    end
    if rem(epoch,50)==0
    feval(@save, ['Rez_' num2str(epoch) '_Aux.mat']);
    end
end

feval(@save, 'RezFinal.mat', 'netG', 'netD');




function [lossG,lossD,gradientsG,gradientsD,stateG,scoreG,scoreD] = ...
    modelLoss(netG,netD,X,T,Z,flipFactor)

% Calculate the predictions for real data with the discriminator network.
YReal = forward(netD,X,T);

% Calculate the predictions for generated data with the discriminator network.
[XGenerated,stateG] = forward(netG,Z,T);
YGenerated = forward(netD,XGenerated,T);

% Calculate probabilities.
probGenerated = sigmoid(YGenerated);
probReal = sigmoid(YReal);

% Calculate the generator and discriminator scores.
scoreG = mean(probGenerated);
scoreD = (mean(probReal) + mean(1-probGenerated)) / 2;

% Flip labels.
numObservations = size(YReal,4);
idx = randperm(numObservations,floor(flipFactor * numObservations));
probReal(:,:,:,idx) = 1 - probReal(:,:,:,idx);

% Calculate the GAN loss.
[lossG, lossD] = ganLoss(probReal,probGenerated);

% For each network, calculate the gradients with respect to the loss.
gradientsG = dlgradient(lossG,netG.Learnables,RetainData=true);
gradientsD = dlgradient(lossD,netD.Learnables);

end

function [lossG, lossD] = ganLoss(scoresReal,scoresGenerated)

% Calculate losses for the discriminator network.
lossGenerated = -mean(log(1 - scoresGenerated));
lossReal = -mean(log(scoresReal));

% Combine the losses for the discriminator network.
lossD = lossReal + lossGenerated;

% Calculate the loss for the generator network.
lossG = -mean(log(scoresGenerated));

end

function [X,T] = preprocessData(XCell,TCell)

% Extract image data from cell and concatenate
X = cat(4,XCell{:});

% Extract label data from cell and concatenate
T = cat(1,TCell{:});

% Rescale the images in the range [-1 1].
X = rescale(X,-1,1,InputMin=0,InputMax=255);

end
