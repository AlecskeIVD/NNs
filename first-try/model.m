%% LOADING DATA
clear
rng("default")
trainsetsize = 500;
points = load("data.mat");

trainx = points.x(1:trainsetsize);
trainy = points.y(1:trainsetsize);
trainclassification = points.classification(1:trainsetsize);

testx = points.x((trainsetsize+1):end);
testy = points.y((trainsetsize+1):end);
testclassification = points.classification((trainsetsize+1):end);

%% INITIALISING MODEL
% I initialise the weights to follow the xavier initialisation, which means
% uniform distributed random weights in the range [-1/sqrt(n); 1/sqrt(n)]
% with n the amount of input nodes
upper = 1/sqrt(2);
lower = -upper;

W2 = lower + (upper-lower) * rand(1, 3);
B2 = lower + (upper-lower) * rand(1, 1);

W1 = lower + (upper-lower) * rand(3, 2);
B1 = lower + (upper-lower) * rand(3, 1);

sigma = @(x) 1 ./ (1 + exp(-x));
yhat = @(x, y) sigma(W2 * sigma(W1 * [x; y] + B1) + B2);


%% GRADIENT DESCENT
alpha = 3;
iterations = 250000;

wrongtest = zeros(iterations, 1);
wrongtraining = zeros(iterations, 1);
for iteration=1:iterations
    totdB1 = [0; 0; 0];
    totdW1 = [0 0; 0 0; 0 0];
    totdB2 = 0;
    totdW2 = [0 0 0];
    for i=1:trainsetsize
        xi = trainx(i);
        yi = trainy(i);
        classi = trainclassification(i);

        [dB1, dW1, dB2, dW2] = calcGradient(xi, yi, classi, B1, W1, B2, W2);
        totdB1 = totdB1 + dB1;
        totdB2 = totdB2 + dB2;
        totdW1 = totdW1 + dW1;
        totdW2 = totdW2 + dW2;

    end
    totdB1 = 1/trainsetsize * (totdB1);
    totdB2 = 1/trainsetsize * (totdB2);
    totdW1 = 1/trainsetsize * (totdW1);
    totdW2 = 1/trainsetsize * (totdW2);

    B1 = B1 - alpha * totdB1;
    B2 = B2 - alpha * totdB2;
    W1 = W1 - alpha * totdW1;
    W2 = W2 - alpha * totdW2;
    yhat = @(x, y) sigma(W2 * sigma(W1 * [x; y] + B1) + B2);
    
    for i=1:trainsetsize
        xi = trainx(i);
        yi = trainy(i);
        classi = trainclassification(i);

        yh = yhat(xi, yi);
        if (yh> 1/2 && classi == 0) || (yh < 1/2 && classi == 1)
            wrongtraining(iteration) = wrongtraining(iteration) + 1;
        end
    end
    for i=1:length(testx)
        xi = testx(i);
        yi = testy(i);
        classi = testclassification(i);

        yh = yhat(xi, yi);
        if (yh> 1/2 && classi == 0) || (yh < 1/2 && classi == 1)
            wrongtest(iteration) = wrongtest(iteration) + 1;
        end
    end

end
%% PLOT DATA
plot(wrongtraining)
hold on
plot(wrongtest)
legend('Number of trainingpoints wrong after each iteration','Number of testpoints wrong after each iteration' )
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
        if yh_val < 0.5
            colorGrid(i, j) = 1; % Red for values < 0.5
        else
            colorGrid(i, j) = 2; % Blue for values >= 0.5
        end
    end
end

% Plot the result using pcolor or imagesc
figure;
hold on;
% Use imagesc to display the color grid
imagesc([0 1], [0 1], colorGrid);

% Adjust colormap: 1 -> Red, 2 -> Blue
colormap([1 0.5 0; 0 0 1]); % Red for 1, Blue for 2

% Set axis properties
axis xy; % Ensure correct orientation
axis equal; % Equal scaling of axes
xlim([0 1]);
ylim([0 1]);
xlabel('x');
ylabel('y');
title('Plot of yhat after 25.000 iterations');

hold off;
