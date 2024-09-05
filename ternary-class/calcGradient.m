function [dB1, dW1, dB2, dW2, dB3, dW3, dB4, dW4] = calcGradient(xi, yi, bluei, orangei, greeni, B1, W1, B2, W2, B3, W3, B4, W4)
sigma = @(x) 1 ./ (1 + exp(-x));
x1 = sigma(W1 * [xi; yi] + B1);
x2 = sigma(W2 * x1 + B2);
x3 = sigma(W3 * x2 + B3);
zout = W4 * x3 + B4;
yhat = softmax(zout);
k = bluei+2*orangei+3*greeni;
dB4 = yhat;
dB4(k) = yhat(k)-1;
dW4 = dB4 * x3';

dB3 = sum(dW4 .* W4, 1)' .* (1-x3);
dW3 = dB3 * x2';

dB2 = sum(dW3 .* W3, 1)' .* (1-x2);
dW2 = dB2 * x1';

dB1 = sum(dW2 .* W2, 1)' .* (1-x1);
dW1 = dB1 * [xi yi];





end