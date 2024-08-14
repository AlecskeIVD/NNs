function [dB1, dW1, dB2, dW2] = calcGradient(xi, yi, classi, B1, W1, B2, W2)
sigma = @(x) 1 ./ (1 + exp(-x));
x1 = sigma(W1 * [xi; yi] + B1);
yhat = sigma(W2 * x1 + B2);
dB2 = (yhat-classi)*yhat*(1-yhat);
dW2 = dB2 * x1';
dB1 = dW2' .* W2' .* (1-x1);
dW1 = [dB1.*xi dB1.*yi];
end