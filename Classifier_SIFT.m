function [SVM_Struct, Features, TrainningResult, TestResult, TrainningSet, TestSet] = Classifier_SIFT(Iter_CrossVal)

Start_CrossVal = 180*(Iter_CrossVal-1)+1;
End_CrossVal = 180*Iter_CrossVal;
Animals = {'dolphin','giraffe','rabbit','sheep'};

load('KmeansCentroid.mat');

% FeaturesPoints = [];
% for ii = 1:4
%     ii
%     for jj = 1:900
%         if jj<Start_CrossVal || jj>End_CrossVal
%             NumOfImages = 10000+jj;
%             Path = [pwd '\FeaturePoints\SIFT\' strjoin(Animals(ii)) '\'  num2str(NumOfImages) '.mat'];
%             Struct = load(Path);
%             StructName = fieldnames(Struct);
%             FeaturesPoints = [FeaturesPoints; getfield(Struct,strjoin(StructName))'];
%         end
%     end
% end
% 
% % first try, k = sqrt(n) = 388
% tic
K = 59;
% FeaturesPoints = single(FeaturesPoints);
% [KmeansCentroid,KmeansIndex] = vl_kmeans(FeaturesPoints', K);
% toc

%convert to histgoram vectors
Features = zeros(900,K,4);
for ii = 1:4
    for jj = 1:900        
        NumOfImages = 10000+jj;
        Path = [pwd '\FeaturePoints\SIFT\' strjoin(Animals(ii)) '\'  num2str(NumOfImages) '_400.mat'];
        Struct = load(Path);
        StructName = fieldnames(Struct);
        FeaturesPoints_temp = single(getfield(Struct,strjoin(StructName))');
        [~, Features_temp] =  min(dist(FeaturesPoints_temp,KmeansCentroid),[],2);
        Features_temp =  hist(Features_temp,K)/length(Features_temp);
        Features(jj,:,ii) = Features_temp;
    end
end

% divde into 5 parts for cross validation
TrainningSet = zeros(2880,K);
TestSet = zeros(720,K);
for ii = 1:4
    Feature_temp = Features(:,:,ii);
    TestSet(180*(ii-1)+1:180*ii,:) = Feature_temp(Start_CrossVal:End_CrossVal,:);
    Feature_temp(Start_CrossVal:End_CrossVal,:) = [];
    TrainningSet(720*(ii-1)+1:720*ii,:) = Feature_temp;
end
Label = [ones(720,1);2*ones(720,1);3*ones(720,1);4*ones(720,1);];

% SVM
SVM_Struct = fitcecoc(TrainningSet,Label,'Coding','onevsall');
TrainningResult = predict(SVM_Struct,TrainningSet);
TestResult = predict(SVM_Struct,TestSet);
ResultLabel = [ones(180,1);2*ones(180,1);3*ones(180,1);4*ones(180,1);];
temp = (TestResult==ResultLabel);
Accuracy = sum(TestResult==ResultLabel)/720
% sum(temp(1:180))/180
% sum(temp(181:360))/180
% sum(temp(361:540))/180
% sum(temp(541:720))/180

