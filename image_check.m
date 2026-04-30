clear all
clc

filename = './test_images/CC7_20x_1_C2_red.png';
% filename = './test_images/ACME_1_3.png';
a = imread(filename);
% I = double(a);
a1 = a(:,:,1);
a2 = a(:,:,2);
a3 = a(:,:,3);

[row, col] = find(a1 ~= 255);

if isequal(a1, a2, a3)
    disp('a1, a2, and a3 are equal')
else
    disp('They are not all equal')
end

%% 
n = 256;
redmap = [linspace(0,1,n)', zeros(n,1), zeros(n,1)];

figure;
imagesc(I);
axis image off;
colormap(gca, redmap);
colorbar;
clim([0 255]);
title("8-bit intensity shown in red");