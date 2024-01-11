close all, 
clear all, 
clc

%% Parametri
% set date
ratioVal=0.15;  % procent  imagini validare
ratioTest=0.15; % procent imagini testare

% antrenare 
NEP=1000; % numar epoci
MBS=20; % numar exempel in minilot (o iteratie)

% arhitectura
NH=40;%numar neuroni ascunsi

%% Fisiere rezultate
filenameRez=['RezMLP_',num2str(NH),'_',num2str(NEP),'.mat'];
filenameDiary=['RezMLP_',num2str(NH),'_',num2str(NEP),'.txt'];

diary(filenameDiary)

%% Set date 
% set date original
% syntheticDir   = fullfile(toolboxdir('vision'),'visiondata','digits','synthetic');
handwrittenDir = fullfile(toolboxdir('vision'),'visiondata','digits','handwritten');

% syntheticDir='D:\Lavinia\licenteDisertatii\Intro\digits';

% creare obiect datastore
imds = imageDatastore(handwrittenDir,'IncludeSubfolders',true,'LabelSource','foldernames');

% impartire set date pe antrenare, validare, testare
[imdsTest,imdsVal,imdsTrain] = splitEachLabel(imds,ratioTest,ratioVal,1-ratioTest-ratioVal,'randomized');

% prepocesare imagini - format novele de gri, redimensionare - vector
imgSize = size(read(imds));

targetSize = [1 imgSize(1)*imgSize(2)]; % 1 exemplu = vector linie

imdsValResized = transform(imdsVal,@(x) reshape(rgb2gray(x),targetSize));
imdsTrainResized = transform(imdsTrain,@(x) reshape(rgb2gray(x),targetSize));
imdsTestResized = transform(imdsTest,@(x) reshape(rgb2gray(x),targetSize));

imdsTestL=arrayDatastore(imdsTest.Labels);
imdsTrainL=arrayDatastore(imdsTrain.Labels);
imdsValL=arrayDatastore(imdsVal.Labels);

imdsValResizedFinal=combine(imdsValResized ,imdsValL);
imdsTrainResizedFinal=combine(imdsTrainResized ,imdsTrainL);
imdsTestResizedFinal=combine(imdsTestResized ,imdsTestL);

imgAfterTransform = read(imdsValResized);
% figure
% imshow(imgAfterTransform,[])


%% MLP - arhitectura
labels=unique(imds.Labels);
numClasses=length(labels);

layers = [
   imageInputLayer(targetSize)
    fullyConnectedLayer(NH )
    tanhLayer
    fullyConnectedLayer(NH) % ponderi initializate cu valori aletoare
    tanhLayer    
    fullyConnectedLayer(numClasses) % ponderi initializate cu valori aletoare
    tanhLayer    
    softmaxLayer 
    classificationLayer];

%% MLP - antrenare
options = trainingOptions('sgdm', ...
    'MiniBatchSize',MBS,...            
    'MaxEpochs',NEP, ...      
    'InitialLearnRate',1e-4, ... 
    'ValidationData',imdsValResizedFinal,...
    'Verbose',false, ...
    'Plots','training-progress');

net=trainNetwork(imdsTrainResizedFinal,layers,options);

%% Rezultate - antrenare
YTrain_net=classify(net,imdsTrainResized );
AccTrain=mean(YTrain_net== imdsTrain.Labels)

%% Rezultate - testare
YTest_net=classify(net,imdsTestResized);
AccTest=mean(YTest_net==imdsTest.Labels)

%% Rezultate - validare
YVal_net=classify(net,imdsValResized);
AccVal=mean(YTest_net==imdsVal.Labels)

%% Rezultate - salvare
feval(@save,filenameRez,'net');
diary off
