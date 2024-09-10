function output = randtranslate(untranslated, maxfactor)
horizontal = randi([-maxfactor, maxfactor]); vertical = randi([-maxfactor, maxfactor]);
if vertical < 0
    output = [zeros(-vertical, size(untranslated, 2)); untranslated(1:(size(untranslated, 1)+vertical),:)];
else
    output = [untranslated((1+vertical):end,:);zeros(vertical, size(untranslated, 2))];
end
if horizontal < 0
    output = [output(:, (1-horizontal):end) zeros(size(untranslated, 1), -horizontal)];
else
    output = [zeros(size(untranslated, 1), horizontal) output(:, 1:(size(untranslated, 2)-horizontal))];
end
end
