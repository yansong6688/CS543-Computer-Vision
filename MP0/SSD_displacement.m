function [ min_x1, min_y1, min_x2, min_y2 ] = SSD_displacement( ch1, ch2, ch3, DISPLACE)

height = size(ch1, 1);
width = size(ch1, 2);
min_SSD1 = Inf;
min_SSD2 = Inf;

for x = -DISPLACE:DISPLACE
    for y = -DISPLACE:DISPLACE
        union13 = circshift(ch1, [y, x]) - ch3;
        union23 = circshift(ch2, [y, x]) - ch3;
        crop13 = union13(floor(0.1*height):floor(0.9*height), floor(0.1*width):floor(0.9*width));
        crop23 = union23(floor(0.1*height):floor(0.9*height), floor(0.1*width):floor(0.9*width));
        SSD1 = sum(sum(crop13.^2));
        SSD2 = sum(sum(crop23.^2));
        if SSD1 < min_SSD1
            min_SSD1 = SSD1;
            min_x1 = x;
            min_y1 = y;
        end
        if SSD2 < min_SSD2
            min_SSD2 = SSD2;
            min_x2 = x;
            min_y2 = y;
        end
    end
end

end

