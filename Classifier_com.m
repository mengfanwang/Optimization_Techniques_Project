clc;clear;
Iter_CrossVal = 5;

[~, Features_SIFT] = Classifier_SIFT(Iter_CrossVal);
[~, Features_LBP] = Classifier_LBP(Iter_CrossVal);
[~, Features_HOG] = Classifier_HOG(Iter_CrossVal);

Features = zeros(900,177,4);
Features(:,1:59,:) = Features_SIFT;
Features(:,60:118,:) = Features_LBP;
Features(:,119:177,:) = Features_HOG;

% divde into 5 parts for cross validation
Start_CrossVal = 180*(Iter_CrossVal-1)+1;
End_CrossVal = 180*Iter_CrossVal;
TrainningSet = zeros(2880,177);
TestSet = zeros(720,177);
for ii = 1:4
    Feature_temp = Features(:,:,ii);
    TestSet(180*(ii-1)+1:180*ii,:) = Feature_temp(Start_CrossVal:End_CrossVal,:);
    Feature_temp(Start_CrossVal:End_CrossVal,:) = [];
    TrainningSet(720*(ii-1)+1:720*ii,:) = Feature_temp;
end
Label = [ones(720,1);2*ones(720,1);3*ones(720,1);4*ones(720,1);];

% SVM
SVM_Struct = fitcecoc(TrainningSet,Label,'Coding','onevsall');
Result = predict(SVM_Struct,TestSet);
ResultLabel = [ones(180,1);2*ones(180,1);3*ones(180,1);4*ones(180,1);];
temp = (Result==ResultLabel);
Accuracy = sum(Result==ResultLabel)/720;
sum(temp(1:180))/180
sum(temp(181:360))/180
sum(temp(361:540))/180
sum(temp(541:720))/180