%% This script will generate easier data for a binary classification problem in 2 dimensions.
% Since the easy network had trouble to capture the complex shape of the
% original data, i will create a simpler datashape based on only one
% gauss-curve
clear all
rng('default')
numberOfRows = 25;
numberOfColumns = 25;
numberOfPoints = numberOfRows * numberOfColumns;
sigma = 0.0625/2;

% The points we want to classify will be in the unit square [0;1]x[0;1] 
a = 0;
b = 1;
c = 0;
d = 1;

X = linspace(a, b, numberOfColumns);
Y = linspace(c, d, numberOfRows);

% The data will be stored in a numberOfPoints x 3 matrix: x, y,
% classification
data = zeros(numberOfPoints, 3);

% The function that will be used for classification. We will label
% something as 1 if it is above the function
function1 = @(x) exp(-(4*x-2).^2);

for i=1:numberOfColumns
    for j=1:numberOfRows
        x = X(i);
        y = Y(j);

        if function1(x) < y 
            classification = 1;
        else
            classification = 0;
        end
        
        % We also add a bit of noise to the data points to simulate
        % measurement errors
        data((i-1)*numberOfRows+j, 1) = x+normrnd(0, sigma);
        data((i-1)*numberOfRows+j, 2) = y+normrnd(0, sigma);
        data((i-1)*numberOfRows+j, 3) = classification;
    end
end
%% Plot to see if the data got generated correctly
fplot(function1, [0 1], 'r', 'LineWidth', 3)
hold on
axis square
xlim([0 1])
ylim([0 1])

positiveRows = data(:, 3) == 1;
negativeRows = data(:, 3) == 0;

scatter(data(positiveRows, 1), data(positiveRows, 2), 'blue', 'filled')
scatter(data(negativeRows, 1), data(negativeRows, 2), 'filled', 'MarkerFaceColor', "#EDB120")
hold off

%% Saving data to .mat file


% Permutate data
permutedIndices = randperm(numberOfPoints);
permutedData = data(permutedIndices, :);
%%
x = permutedData(:, 1);
y = permutedData(:, 2);
classification = permutedData(:, 3);

save('easierdata.mat', 'x', 'y', 'classification')



