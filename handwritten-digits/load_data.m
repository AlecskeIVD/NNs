function [imagestrain, labelstrain, imagestest, labelstest] = load_data()
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


numTrainFiles = 750;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainFiles,'randomize');

imagestrain = zeros(28*28, numTrainFiles*10);
labelstrain = zeros(10, numTrainFiles*10);

imagestest = zeros(28*28, 10000-numTrainFiles*10);
labelstest = zeros(10, 10000-numTrainFiles*10);

for imageindex=1:7500
    imagestrain(:, imageindex) = reshape(add_noise(randtranslate(double(readimage(imdsTrain,imageindex)), 3), 0.03), [], 1)/255;
    labelstrain(double(imdsTrain.Labels(imageindex)), imageindex) = 1; % A 1 in the i'th element corresponds to a drawing of i-1 i.e. A 1 in the first element means label 0.
end

for imageindex=1:2500
    imagestest(:, imageindex) = double(reshape(readimage(imdsTest,imageindex), [], 1))/255;
    labelstest(double(imdsTest.Labels(imageindex)), imageindex) = 1; % A 1 in the i'th element corresponds to a drawing of i-1 i.e. A 1 in the first element means label 0.
end

%temp = randperm(numTrainFiles*10, 7500);
%imagestrain = imagestrain(:, temp);
%labelstrain = labelstrain(:, temp);

%temp = randperm(10000-numTrainFiles*10, 2500);
%imagestest = imagestest(:, temp);
%labelstest = labelstest(:, temp);
end

