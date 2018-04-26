clc;clear;close all;
m = zeros(900,59);
for ii = 1:900
    NumOfImages = 10000+ii;
    Im = rgb2gray(imread(['JPEGImages\sheep\sheep_',num2str(NumOfImages),'.jpg']));
    m(ii,:) =  extractLBPFeatures(Im);
    

%     save(['FeaturePoints\LBP\dolphin\',num2str(NumOfImages),'.mat'],'Points');

    ii
end

