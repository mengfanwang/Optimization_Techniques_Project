clc;clear;
Iter_CrossVal = 1;

[SVM_Struct_SIFT, Features_SIFT, TrainningResult_SIFT, TestResult_SIFT] = Classifier_SIFT(Iter_CrossVal);
[SVM_Struct_LBP, Features_LBP, TrainningResult_LBP, TestResult_LBP] = Classifier_LBP(Iter_CrossVal);
[SVM_Struct_HOG, Features_HOG, TrainningResult_HOG, TestResult_HOG] = Classifier_HOG(Iter_CrossVal);

Label = [ones(720,1);2*ones(720,1);3*ones(720,1);4*ones(720,1);];
Weight = ones(2880,1)/2880;

Error_SIFT = sum(Weight.*(TrainningResult_SIFT~=Label));
a_SIFT = log((1-Error_SIFT)/Error_SIFT)/2;
for ii = 1:2880
    if TrainningResult_SIFT(ii) == Label(ii)
        Weight(ii) = Weight(ii)* exp(-a_SIFT);
    else
        Weight(ii) = Weight(ii)* exp(a_SIFT);
    end
end
Weight = Weight/sum(Weight);

Error_LBP = sum(Weight.*(TrainningResult_LBP~=Label));
a_LBP = log((1-Error_LBP)/Error_LBP)/2;
for ii = 1:2880
    if TrainningResult_LBP(ii) == Label(ii)
        Weight(ii) = Weight(ii)* exp(-a_LBP);
    else
        Weight(ii) = Weight(ii)* exp(a_LBP);
    end
end
Weight = Weight/sum(Weight);

Error_HOG = sum(Weight.*(TrainningResult_HOG~=Label));
a_HOG = log((1-Error_HOG)/Error_HOG)/2;

Result_mat = zeros(720,4);
for ii = 1:720
    Result_mat(ii,TestResult_SIFT(ii)) = Result_mat(TestResult_SIFT(ii))+a_SIFT;
    Result_mat(ii,TestResult_LBP(ii)) = Result_mat(TestResult_LBP(ii))+a_LBP;
    Result_mat(ii,TestResult_HOG(ii)) = Result_mat(TestResult_HOG(ii))+a_HOG;
end
[~, TestResult] =  max(Result_mat,[],2);
ResultLabel = [ones(180,1);2*ones(180,1);3*ones(180,1);4*ones(180,1);];
temp = (TestResult==ResultLabel);
Accuracy = sum(TestResult==ResultLabel)/720;
% sum(temp(1:180))/180
% sum(temp(181:360))/180
% sum(temp(361:540))/180
% sum(temp(541:720))/180

