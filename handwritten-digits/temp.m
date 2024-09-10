for i = 1:1
    index = randi(7500); % Random index within the range of training samples
    img = reshape(imagestrain(:, index), 28, 28); % Reshape the flattened image back to 28x28
    label = find(labelstrain(:, index) == 1) - 1; % Find the index of the 1 in one-hot encoding (subtract 1 for label 0-9)
    
    % Display the image and its corresponding label
    figure;
    imshow(img);
    title(['Label: ', num2str(label)]);
    
    figure
    imshow(randtranslate(img, 5));
end