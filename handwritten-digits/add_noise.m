function output = add_noise(input, noise_factor)
number_of_noise_points = round(noise_factor*size(input, 1)*size(input, 2));
output = input(:, :);
for i=1:number_of_noise_points
    m = randi(size(input, 1)); n=randi(size(input, 2));
    output(m, n) = rand(1);
end
end
