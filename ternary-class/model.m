%% LOADING DATA
clear
rng("default")
trainsetsize = 2000;
points = load("data.mat");

trainx = points.x(1:trainsetsize);
trainy = points.y(1:trainsetsize);
trainblue = points.blue(1:trainsetsize);
trainorange = points.orange(1:trainsetsize);
traingreen = points.green(1:trainsetsize);

testx = points.x((trainsetsize+1):end);
testy = points.y((trainsetsize+1):end);
testblue = points.blue((trainsetsize+1):end);
testorange = points.orange((trainsetsize+1):end);
testgreen = points.green((trainsetsize+1):end);
testsetsize = length(testx);

%% INITIALISING MODEL
% I initialise the weights to follow the xavier initialisation, which means
% uniform distributed random weights in the range [-1/sqrt(n); 1/sqrt(n)]
% with n the amount of input nodes
upper = sqrt(6) / sqrt(2 + 3);
lower = -upper;

W4 = lower + (upper-lower) * rand(3, 3);
B4 = lower + (upper-lower) * rand(3, 1);

W3 = lower + (upper-lower) * rand(3, 3);
B3 = lower + (upper-lower) * rand(3, 1);

W2 = lower + (upper-lower) * rand(3, 5);
B2 = lower + (upper-lower) * rand(3, 1);

W1 = lower + (upper-lower) * rand(5, 2);
B1 = lower + (upper-lower) * rand(5, 1);

sigma = @(x) 1 ./ (1 + exp(-x));
zout = @(x, y) (W4*sigma(W3*sigma(W2 * sigma(W1 * [x; y] + B1) + B2)+B3)+B4);
yhat = @(x, y) softmax(zout(x, y));

%% GRADIENT DESCENT
alpha = 0.3;
epochs = 50000;
epochsize = trainsetsize;

% I will divide the dataset into K smaller datasets to implement stochastic
% gradient descent
K = 10;
subsize = epochsize/K;

correcttest = zeros(epochs, 1);
correcttraining = zeros(epochs, 1);
for epoch=1:epochs
    % Split up data in k random minidatasets
    indices = randperm(epochsize);

    for iteration=1:K
        trainindices = indices((iteration-1)*subsize+1: iteration*subsize);
        subtrainx = trainx(trainindices);
        subtrainy = trainy(trainindices);
        subtrainblue = trainblue(trainindices);
        subtrainorange = trainorange(trainindices);
        subtraingreen = traingreen(trainindices);

        % Calculate gradient and change parameters
        totdW4 = zeros(3, 3);
        totdB4 = zeros(3, 1);

        totdW3 = zeros(3, 3);
        totdB3 = zeros(3, 1);

        totdW2 = zeros(3, 5);
        totdB2 = zeros(3, 1);

        totdW1 = zeros(5, 2);
        totdB1 = zeros(5, 1);
        for i=1:subsize
            xi = subtrainx(i);
            yi = subtrainy(i);
            bluei = subtrainblue(i);
            orangei = subtrainorange(i);
            greeni = subtraingreen(i);

            [dB1, dW1, dB2, dW2, dB3, dW3, dB4, dW4] = calcGradient(xi, yi, bluei, orangei, greeni, B1, W1, B2, W2, B3, W3, B4, W4);
            totdB1 = totdB1 + dB1;
            totdB2 = totdB2 + dB2;
            totdW1 = totdW1 + dW1;
            totdW2 = totdW2 + dW2;
            totdB3 = totdB3 + dB3;
            totdB4 = totdB4 + dB4;
            totdW3 = totdW3 + dW3;
            totdW4 = totdW4 + dW4;

        end
        totdB1 = 1/subsize * (totdB1);
        totdB2 = 1/subsize * (totdB2);
        totdB3 = 1/subsize * (totdB3);
        totdB4 = 1/subsize * (totdB4);
        totdW1 = 1/subsize * (totdW1);
        totdW2 = 1/subsize * (totdW2);
        totdW3 = 1/subsize * (totdW3);
        totdW4 = 1/subsize * (totdW4);


        B1 = B1 - alpha * totdB1;
        B2 = B2 - alpha * totdB2;
        W1 = W1 - alpha * totdW1;
        W2 = W2 - alpha * totdW2;
        B3 = B3 - alpha * totdB3;
        B4 = B4 - alpha * totdB4;
        W3 = W3 - alpha * totdW3;
        W4 = W4 - alpha * totdW4;
        
    end
    zout = @(x, y) (W4*sigma(W3*sigma(W2 * sigma(W1 * [x; y] + B1) + B2)+B3)+B4);
    yhat = @(x, y) softmax(zout(x, y));
    for i=1:trainsetsize
        xi = trainx(i);
        yi = trainy(i);
        bluei = trainblue(i);
        orangei = trainorange(i);
        greeni = traingreen(i);

        yh = yhat(xi, yi);
        [~, class] = max(yh);
        if (class == 1 && bluei == 1) || (class == 2 && orangei == 1) || (class == 3 && greeni == 1)
            correcttraining(epoch) = correcttraining(epoch) + 1/trainsetsize;
        end
    end
    for i=1:testsetsize
        xi = testx(i);
        yi = testy(i);
        bluei = testblue(i);
        orangei = testorange(i);
        greeni = testgreen(i);

        yh = yhat(xi, yi);
        [~, class] = max(yh);
        if (class == 1 && bluei == 1) || (class == 2 && orangei == 1) || (class == 3 && greeni == 1)
            correcttest(epoch) = correcttest(epoch) + 1/testsetsize;
        end
    end

end

%% PLOT DATA
plot(correcttraining)
hold on
plot(correcttest)
legend('Percentage of trainingpoints correct after each epoch','Percentage of testpoints wrong after each epoch' )
hold off

%% PLOT UNIT SQUARE
% Define grid resolution
resolution = 1000;

% Create grid over [0, 1] x [0, 1]
[x_grid, y_grid] = meshgrid(linspace(0, 1, resolution), linspace(0, 1, resolution));

% Initialize a matrix to hold the colors
colorGrid = zeros(resolution, resolution);

% Evaluate yhat at each point in the grid
for i = 1:resolution
    for j = 1:resolution
        yh_val = yhat(x_grid(i, j), y_grid(i, j));
        [~,class] = max(yh_val);
        if class == 1
            colorGrid(i, j) = 1; % Blue
        elseif class == 2
            colorGrid(i, j) = 2; % Orange
        else
            colorGrid(i, j) = 3; % Green
        end
    end
end

% Plot the result using pcolor or imagesc
figure;
hold on;
% Use imagesc to display the color grid
imagesc([0 1], [0 1], colorGrid);

% Adjust colormap: 1 -> Blue, 2 -> Orange, 3 -> Green
colormap([0 0 1; 1 0.5 0; 0 1 0]);

% Set axis properties
axis xy; % Ensure correct orientation
axis equal; % Equal scaling of axes
xlim([0 1]);
ylim([0 1]);
xlabel('x');
ylabel('y');
title('Plot of yhat after 50.000 epochs');

hold off;

