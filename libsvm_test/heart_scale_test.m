load heart_scale
[N D] = size(heart_scale_inst);

% Determine the train and test index,select top 200 as training data
% else as test data
trainIndex = zeros(N,1);
trainIndex(1:200) = 1;
testIndex = zeros(N,1);
testIndex(201:N) = 1;
trainData = heart_scale_inst(trainIndex==1,:);
trainLabel = heart_scale_label(trainIndex==1,:);
testData = heart_scale_inst(testIndex==1,:);
testLabel = heart_scale_label(testIndex==1,:);

% Train the SVM
model = svmtrain(trainLabel, trainData, '-c 1 -g 0.07 -b 1');
% Use the SVM model to classify the data
[predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model, '-b 1'); % run the SVM model on the test data