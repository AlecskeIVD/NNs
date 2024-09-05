%% This script will generate the data for a ternary classification problem in 2 dimensions.
clear all
rng('default')
numberOfRows = 25*2;
numberOfColumns = 25*2;
numberOfPoints = numberOfRows * numberOfColumns;
sigma = 0.0625/2;

% The points we want to classify will be in the unit square [0;1]x[0;1] 
a = 0;
b = 1;
c = 0;
d = 1;

X = linspace(a, b, numberOfColumns);
Y = linspace(c, d, numberOfRows);

% The data will be stored in a numberOfPoints x 5 matrix: x, y,
% blue, orange, green
data = zeros(numberOfPoints, 5);


for i=1:numberOfColumns
    for j=1:numberOfRows
        x = X(i);
        y = Y(j);

        if (x-1/2)^2 + (y-1/2)^2 < (1/3)^2
            classification = [1 0 0];
        elseif y > 1/2
            classification = [0 1 0];
        else
            classification = [0 0 1];
        end
        
        % We also add a bit of noise to the data points to simulate
        % measurement errors
        data((i-1)*numberOfRows+j, 1) = x+normrnd(0, sigma);
        data((i-1)*numberOfRows+j, 2) = y+normrnd(0, sigma);
        data((i-1)*numberOfRows+j, 3:5) = classification;
    end
end
%% Plot to see if the data got generated correctly

blueRows = data(:, 3) == 1;
orangeRows = data(:, 4) == 1;
greenRows = data(:, 5) == 1;

scatter(data(blueRows, 1), data(blueRows, 2), 'blue', 'filled')
hold on
axis square
xlim([0 1])
ylim([0 1])
scatter(data(orangeRows, 1), data(orangeRows, 2), 'filled', 'MarkerFaceColor', "#EDB120")
scatter(data(greenRows, 1), data(greenRows, 2), 'filled', 'MarkerFaceColor', '#00FF00')
hold off

%% Saving data to .mat file


% Permutate data
permutedIndices = randperm(numberOfPoints);
permutedData = data(permutedIndices, :);
%%
x = permutedData(:, 1);
y = permutedData(:, 2);
blue = permutedData(:, 3);
orange = permutedData(:, 4);
green = permutedData(:, 5);

save('data.mat', 'x', 'y', 'blue', 'orange', 'green')



