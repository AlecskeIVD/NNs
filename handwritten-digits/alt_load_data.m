function [imagestrain, labelstrain, imagestest, labelstest] = alt_load_data()
temp = load("mnist.mat", "test", "training");
test = temp.test;
training = temp.training;

imagestrain = reshape(training.images, 28*28, training.count);
labelstrain = zeros(10, training.count);

for i=1:training.count
    labelstrain(training.labels(i)+1, i) = 1;
end

imagestest = reshape(test.images, 28*28, test.count);
labelstest = zeros(10, test.count);

for i=1:test.count
    labelstest(test.labels(i)+1, i) = 1;
end







