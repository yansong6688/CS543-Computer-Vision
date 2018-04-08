function [ min_x1, min_y1, min_x2, min_y2] = SSD_multiscale(ch1, ch2, ch3)

height = size(ch1, 1);
width = size(ch1, 2);

if height < 500 && width < 500
    [min_x1, min_y1, min_x2, min_y2] = SSD_displacement(ch1, ch2, ch3, 30);
else
    resized_ch1 = imresize(ch1, 0.5);
    resized_ch2 = imresize(ch2, 0.5);
    resized_ch3 = imresize(ch3, 0.5);
    [x11, y11, x21, y21] = SSD_multiscale(resized_ch1, resized_ch2, resized_ch3);
    shift_ch1 = circshift(ch1, 2*[x11, y11]);
    shift_ch2 = circshift(ch2, 2*[x21, y21]);
    [x12, y12, x22, y22] = SSD_displacement(shift_ch1, shift_ch2, ch3, 1);
    min_x1 = 2 * x11 + x12;
    min_y1 = 2 * y11 + y12;
    min_x2 = 2 * x21 + x22;
    min_y2 = 2 * y21 + y22;
end

end

