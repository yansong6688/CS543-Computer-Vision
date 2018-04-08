function [ max_x1, max_y1, max_x2, max_y2 ] = NCC_displacement(ch1, ch2, ch3, DISPLACE)

height = size(ch3, 1);
width = size(ch3, 2);

max_NCC1 = -Inf;
max_NCC2 = -Inf;

tic
for x = -DISPLACE:DISPLACE
    for y = -DISPLACE:DISPLACE
        shift1 = circshift(ch1, [y, x]);
        shift2 = circshift(ch2, [y, x]);
        crop1 = shift1(floor(0.1*height):floor(0.9*height), floor(0.1*width):floor(0.9*width));
        crop2 = shift2(floor(0.1*height):floor(0.9*height), floor(0.1*width):floor(0.9*width));
        crop3 = ch3(floor(0.1*height):floor(0.9*height), floor(0.1*width):floor(0.9*width));
        mean1 = crop1 - mean(crop1(:));
        mean2 = crop2 - mean(crop2(:));
        mean3 = crop3 - mean(crop3(:));
        std1 = mean1 ./ std(mean1(:), 0, 1);
        std2 = mean2 ./ std(mean2(:), 0, 1);
        std3 = mean3 ./ std(mean3(:), 0, 1);
        
        NCC1 = sum(dot(std3, std1));
        NCC2 = sum(dot(std3, std2));
        if NCC1 > max_NCC1
            max_NCC1 = NCC1;
            max_x1 = x;
            max_y1 = y;
        end
        if NCC2 > max_NCC2
            max_NCC2 = NCC2;
            max_x2 = x;
            max_y2 = y;
        end
    end
end
toc


end

