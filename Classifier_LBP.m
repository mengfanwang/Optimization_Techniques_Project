function [SVM_Struct, Features, TrainningResult, TestResult, TrainningSet, TestSet] = Classifier_LBP(Iter_CrossVal)

%import data
Animals = {'dolphin','giraffe','rabbit','sheep'};
Features = zeros(900,59,4);
for ii = 1:4
    Path = [pwd '\FeaturePoints\LBP\' strjoin(Animals(ii)) '\' strjoin(Animals(ii)) '_LBP.mat'];
    Struct = load(Path);
    StructName = fieldnames(Struct);
    Features(:,:,ii) = getfield(Struct,strjoin(StructName));
end

% divde into 5 parts for cross validation
Start_CrossVal = 180*(Iter_CrossVal-1)+1;
End_CrossVal = 180*Iter_CrossVal;
TrainningSet = zeros(2880,59);
TestSet = zeros(720,59);
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
