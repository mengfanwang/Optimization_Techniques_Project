clc;clear;close all;
m = zeros(900,1);
for ii = 308:900
    NumOfImages = 10000+ii;
    Im = rgb2gray(imread(['JPEGImages\rabbit\rabbit_',num2str(NumOfImages),'.jpg']));
%     figure(ii),imshow(Im),hold on;
    Im = single(Im);
    
    
    % find the best threshold
    Threshold = 0;
    NumOfPoints = inf;
    
    [~,Points] = vl_sift(Im,'PeakThresh',Threshold);
    NumOfPoints = size(Points,2);
    while NumOfPoints > 400
    Threshold = Threshold + 0.5;
    [~,Points] = vl_sift(Im,'PeakThresh',Threshold);
    NumOfPoints = size(Points,2);
    end

    
    m(ii) = size(Points,2);

    save(['FeaturePoints\SIFT\rabbit\',num2str(NumOfImages),'_400.mat'],'Points');

    NumOfPoints
    ii
end
m
