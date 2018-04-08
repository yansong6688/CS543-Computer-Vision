clear all
clc
figure(1)
img = imread('data/01112v.jpg');
imagesc(img);
colormap('gray');
truesize;
img = im2double(img);
height = floor(size(img, 1) / 3);
width = size(img, 2);
DISPLACE = 20;

B = img(1:height, :);             % B
G = img(height+1: 2*height, :);   % G
R = img(2*height+1:3*height, :);  % R 

figure(2)
imagesc(B);
colormap('gray');
figure(3)
imagesc(G);
colormap('gray');
figure(4)
imagesc(R);
colormap('gray');

%% R as template
[x1, y1, x2, y2] = NCC_displacement(B, G, R, DISPLACE);    % align ch1, ch2 to ch3
output1 = zeros(height, width, 3);
output1(:, :, 1) = R;
output1(:, :, 2) = circshift(G, [y2, x2]);
output1(:, :, 3) = circshift(B, [y1, x1]);
imagesc(output1);
title1 = strcat('NCC: R as template; x1=', int2str(x1), ' y1=', int2str(y1), ' x2=', int2str(x2), ' y2=', int2str(y2));
title(title1);
truesize;

%% G as template
[x1, y1, x2, y2] = NCC_displacement(B, R, G, DISPLACE);    % align ch1, ch2 to ch3
output2 = zeros(height, width, 3);
output2(:, :, 1) = circshift(R, [y2, x2]);
output2(:, :, 2) = G;
output2(:, :, 3) = circshift(B, [y1, x1]);
imagesc(output2);
title2 = strcat('NCC: G as template; x1=', int2str(x1), ' y1=', int2str(y1), ' x2=', int2str(x2), ' y2=', int2str(y2));
title(title2);
truesize;

%% B as template
[x1, y1, x2, y2] = NCC_displacement(G, R, B, DISPLACE);    % align ch1, ch2 to ch3
output3 = zeros(height, width, 3);
output3(:, :, 1) = circshift(R, [y2, x2]);
output3(:, :, 2) = circshift(G, [y1, x1]);
output3(:, :, 3) = B;
imagesc(output3);
title3 = strcat('NCC: B as template; x1=', int2str(x1), ' y1=', int2str(y1), ' x2=', int2str(x2), ' y2=', int2str(y2));
title(title3);
truesize;
