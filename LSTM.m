clc
clear
data21=xlsread("GGDPdata.xlsx");
GDP1=data21(1:29,12);
datayuce=[GDP1];
out=[];
trainData=datayuce(:,1);
mu = mean(trainData);
sig = std(trainData);
trainDataStandardized = (trainData - mu) / sig;
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
layers = [sequenceInputLayer(1)
            lstmLayer(250)
            fullyConnectedLayer(1)
            regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 120, ...
    'LearnRateDropFactor', 0.25, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');
XTrain=XTrain';
YTrain=YTrain';
net = trainNetwork(XTrain,YTrain,layers,options);
net = predictAndUpdateState(net, XTrain);
[net, YPred] = predictAndUpdateState(net, YTrain(end));
for i = 2:30
    [net, YPred(:, i)] = predictAndUpdateState(net, YPred(:, i-1), 'ExecutionEnvironment', 'cpu');
end
YPredout{1} = sig*YPred + mu;
out=[out YPredout{1}'];
figure(1)
plot(GDP1);
hold on 
plot(30:1:59,out(:,1)');
title('The GGDP prediction in future 30 years');
year = zeros(1,59);
year(1,:) = 1:1:59;
GDPData = zeros(1,29);
GDPData(1,:) = GDP1';
GGDPData = zeros(1,30);
GGDPData(1,:) = out(:,1)';
save("year.mat" , 'year');
save("GDPData.mat" , 'GDPData');
save("GGDPData.mat" , 'GGDPData');