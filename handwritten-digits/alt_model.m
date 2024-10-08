%% LOADING DATA
clear
rng("default")
[imagestrain, labelstrain, imagestest, labelstest] = load_data;
trainsetsize = size(imagestrain, 2);
testsetsize = size(labelstest, 2);


%% INITIALISING MODEL
% I initialise the weights to follow the xavier initialisation, which means
% uniform distributed random weights in the range [-1/sqrt(n); 1/sqrt(n)]
% with n the amount of input nodes
upper = 1/sqrt(28*28);
lower = -upper;

W3 = lower + (upper-lower) * rand(10, 20);
B3 = lower + (upper-lower) * rand(10, 1);

W2 = lower + (upper-lower) * rand(20, 200);
B2 = lower + (upper-lower) * rand(20, 1);

W1 = lower + (upper-lower) * rand(200, 28*28);
B1 = lower + (upper-lower) * rand(200, 1);

sigma = @(x) 1 ./ (1+exp(-x));
zout = @(imagevector) (W3*sigma(W2 * sigma(W1 * imagevector + B1) + B2)+B3);
yhat = @(imagevector) softmax(zout(imagevector));

%% GRADIENT DESCENT
alpha = 0.1;
epochs = 100;
epochsize = trainsetsize;

% I will divide the dataset into K smaller datasets to implement stochastic
% gradient descent
K = 750;
subsize = epochsize/K;

correcttest = zeros(epochs, 1);
correcttraining = zeros(epochs, 1);
for epoch=1:epochs
    % Split up data in k random minidatasets
    disp("epoch: " + string(epoch))
    indices = randperm(epochsize);

    for iteration=1:K
        trainindices = indices((iteration-1)*subsize+1: iteration*subsize);
        subtrainimagevectors = imagestrain(:, trainindices);
        subtrainlabels = labelstrain(:, trainindices);

        % Calculate gradient and change parameters
        totdW3 = zeros(10, 20);
        totdB3 = zeros(10, 1);

        totdW2 = zeros(20, 200);
        totdB2 = zeros(20, 1);

        totdW1 = zeros(200, 28*28);
        totdB1 = zeros(200, 1);
        for i=1:subsize
            imagevectori = subtrainimagevectors(:,i);
            [~, labeli] = max(subtrainlabels(:,i));
            labeli = labeli-1;

            [dB1, dW1, dB2, dW2, dB3, dW3] = alt_calcGradient(imagevectori, labeli, B1, W1, B2, W2, B3, W3);
            totdB1 = totdB1 + dB1;
            totdB2 = totdB2 + dB2;
            totdW1 = totdW1 + dW1;
            totdW2 = totdW2 + dW2;
            totdB3 = totdB3 + dB3;
            totdW3 = totdW3 + dW3;

        end
        totdB1 = 1/subsize * (totdB1);
        totdB2 = 1/subsize * (totdB2);
        totdB3 = 1/subsize * (totdB3);
        totdW1 = 1/subsize * (totdW1);
        totdW2 = 1/subsize * (totdW2);
        totdW3 = 1/subsize * (totdW3);


        B1 = B1 - alpha * totdB1;
        B2 = B2 - alpha * totdB2;
        W1 = W1 - alpha * totdW1;
        W2 = W2 - alpha * totdW2;
        B3 = B3 - alpha * totdB3;
        W3 = W3 - alpha * totdW3;
        
    end
    zout = @(imagevector) (W3*sigma(W2 * sigma(W1 * imagevector + B1) + B2)+B3);
    yhat = @(imagevector) softmax(zout(imagevector));
    for i=1:trainsetsize
        imagevectori = imagestrain(:,i);
        [~, labeli] = max(labelstrain(:,i));
        labeli = labeli-1;

        yh = yhat(imagevectori);
        [~, class] = max(yh);
        class = class-1;
        %disp(yh)
        if (class == labeli)
            correcttraining(epoch) = correcttraining(epoch) + 1/trainsetsize;
        end
    end
    for i=1:testsetsize
        imagevectori = imagestest(:,i);
        [~, labeli] = max(labelstest(:,i));
        labeli = labeli-1;

        yh = yhat(imagevectori);
        [~, class] = max(yh);
        class = class-1;
        if (class == labeli)
            correcttest(epoch) = correcttest(epoch) + 1/testsetsize;
        end
    end

end

%% PLOT DATA
plot(correcttraining)
hold on
plot(correcttest)
legend('Fraction of trainingpoints correct after each epoch','Fraction of testpoints wrong after each epoch' )
xlabel("Epochs")
hold off
%% SAVE WEIGHTS
save('weights.mat', 'B1', 'B2', 'B3', 'W1', 'W2', "W3");

