clc;clear;close all;
m = zeros(900,59);
for ii = 1:900
    NumOfImages = 10000+ii;
    Im = rgb2gray(imread(['JPEGImages\rabbit\rabbit_',num2str(NumOfImages),'.jpg']));
    Features =  extractHOGFeatures(Im);
    Features = hist(Features,59);
    m(ii,:) = Features/sum(Features);
    

%     save(['FeaturePoints\LBP\dolphin\',num2str(NumOfImages),'.mat'],'Points');

    ii
end
m = m/max(max(m));
