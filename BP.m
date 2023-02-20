%% Data Reading and Pre-treating

clc;
clear;

rowIndex = [3,5,9,10,2];
sheetIndex = [1,2,3,4,5,6,7,8];
yearIndex = 1:1:29;
rowLength = length(rowIndex);
sheetLength = length(sheetIndex);
yearLength = length(yearIndex);
a = 0.8;
aVal = [1,a,a,a,a,1-a,1-a,1-a];
sheetName = ["GDP","LMR","LEC","LFR","AFW","CMC","PD","EU","CO2","TEMP"];

inputData = zeros(sheetLength,rowLength*yearLength);
outputData = zeros(2,rowLength*yearLength);
GGDPData = zeros(3,rowLength*yearLength);


for sheet = 1:sheetLength
    tmp = xlsread("data.xlsx",sheetName(sheetIndex(sheet)));
    for row = 1:rowLength
        for year = 1:yearLength
            inputData(sheet,(row-1)*yearLength+year) = tmp(rowIndex(row),yearIndex(year))*aVal(sheet);            
        end        
    end
end

CO2Data = xlsread("data.xlsx",sheetName(9));
tempData = xlsread("data.xlsx",sheetName(10));
for row = 1:rowLength
    for year = 1:yearLength
        outputData(1,(row-1)*yearLength+year) = CO2Data(rowIndex(row),yearIndex(year));
        outputData(2,(row-1)*yearLength+year) = tempData(rowIndex(row),yearIndex(year));
    end
end

year = 1991:1:2019;
titleName = ["CHN","DEU","AUS","ZAF","USA"];
for i = 1:rowLength
    [inputRaw,k{i}] = mapminmax(inputData(:,(i-1)*yearLength+1:i*yearLength),0,1);

    GGDPData(1,(i-1)*yearLength+1:i*yearLength) = inputRaw(1,:) - (a*(inputRaw(2,:)+inputRaw(3,:)+inputRaw(4,:)) - (1-a)*(inputRaw(5,:)+inputRaw(6,:) + inputRaw(7,:) + inputRaw(8,:)))*1;
    [GGDPData(2,(i-1)*yearLength+1:i*yearLength),~] = mapminmax(GGDPData(1,(i-1)*yearLength+1:i*yearLength),0,1);
    temp2 = mapminmax('reverse',GGDPData(2,(i-1)*yearLength+1:i*yearLength),k{i});
    GGDPData(3,(i-1)*yearLength+1:i*yearLength) = temp2(1,:);
end

%% Model setting up, training and testing

rng(0);
inputNum = length(inputData(:,1:116));
P = inputData(:,1:116);
T = outputData(:,1:116);
temp = randperm(inputNum);
inputTest = P(:,temp(round(0.86*inputNum+1):end));
inputTrain = P(:,temp(1:round(0.86*inputNum)));
outputTest = T(:,temp(round(0.86*inputNum+1):end));
outputTrain = T(:,temp(1:round(0.86*inputNum)));
[inputn,inputps] = mapminmax(inputTrain,0,1);
[outputn,outputps] = mapminmax(outputTrain);
inputnTest = mapminmax('apply',inputTest,inputps);
inputnTrain = mapminmax('apply',inputTrain,inputps);

bpNet = newff(inputn,outputn,11);
bpNet.trainParam.epochs = 1000;
bpNet.trainParam.max_fail = 2000;
bpNet.trainParam.lr = 0.02;
bpNet.trainParam.goal = 0.0001;
trainParam.min_grad = 1e-6;
trainParam.min_fail = 10;
bpNet = train(bpNet,inputn,outputn);
an = sim(bpNet,inputnTest);
testSimu = mapminmax('reverse',an,outputps);
error = testSimu-outputTest;
bn = sim(bpNet,inputn);
trainSimuBp = mapminmax('reverse',bn,outputps);


testCO2Data = zeros(3,16);
testCO2Data(1,:) = 1:1:16;
testCO2Data(2,:) = outputTest(1,:);
testCO2Data(3,:) = testSimu(1,:);
save("test1.mat" , 'testCO2Data');

testTempData = zeros(3,16);
testTempData(1,:) = 1:1:16;
testTempData(2,:) = outputTest(2,:);
testTempData(3,:) = testSimu(2,:);
save("test2.mat" , 'testTempData');

trainCO2Data = zeros(3,100);
trainCO2Data(1,:) = 1:1:100;
trainCO2Data(2,:) = outputTrain(1,:);
trainCO2Data(3,:) = trainSimuBp(1,:);
save("train1.mat" , 'trainCO2Data');

trainTempData = zeros(3,100);
trainTempData(1,:) = 1:1:100;
trainTempData(2,:) = outputTrain(2,:);
trainTempData(3,:) = trainSimuBp(2,:);
save("train2.mat" , 'trainTempData');

[outputTest,~] = mapminmax(outputTest);
[testSimu,~] = mapminmax(testSimu);
[R20,MAE0,RMSE0,MAPE0,~,PRD0] = solve(outputTest(1,:),testSimu(1,:));
[R21,MAE1,RMSE1,MAPE1,~,PRD1] = solve(outputTest(2,:),testSimu(2,:));
R2 = (R20+R21)/2;
RMSE = (RMSE0 + RMSE1)/2;
MAE = (MAE0 + MAE1)/2;
MAPE = abs(MAPE0 + MAPE1)/2;
error = testSimu-outputTest;
PRD = (PRD0 + PRD1)/2;

disp(append('R2:',num2str(R2)));
disp(append('Root Mean Square Error(RMSE):',num2str(RMSE)));
disp(append('Mean Absolute Error(MAE):',num2str(MAE)));
disp(append('Mean Absolute Percentage Error(MAPE):',num2str(MAPE*100),'%'));
disp(append('relative analysis error(PRD):',num2str(PRD)));

inputDataGGDP = inputData;
inputDataGGDP(1,:) = GGDPData(2,:);
[inputDataGGDPM,inputpsRE] = mapminmax(inputDataGGDP,0,1);
per2 = sim(bpNet,inputDataGGDPM);
outputDataGGDP = outputData;
[outputDataGGDPM,outputpsRE] = mapminmax(outputDataGGDP,0,1);

%% Data visualization

figure(1)
sgtitle("The environmental impact after GGDP replaced GDP");
outCO2 = zeros(5,yearLength);
outTemp = zeros(5,yearLength);
for i = 1:5
    if i>1
        subplot(2,2,i-1);
        yyaxis left
        plot(outputDataGGDPM(1,(i-1)*yearLength+1:i*yearLength))
        hold on
        plot(per2(1,(i-1)*yearLength+1:i*yearLength))
        yyaxis right
        plot(outputDataGGDPM(2,(i-1)*yearLength+1:i*yearLength))
        hold on
        plot(per2(2,(i-1)*yearLength+1:i*yearLength))
        title(titleName(i));
        legend('CO2 true value','CO2 expected value','temperature true value','temperture expected value');
    end
    outCO2(i,:) = per2(1,(i-1)*yearLength+1:i*yearLength);
    outTemp(i,:) = per2(2,(i-1)*yearLength+1:i*yearLength);
end


figure(2)
predictedCO2Data = zeros(6,29);
predictedCO2Data(1,:) = 1:1:29;
for i = 1:5
    plot(outCO2(i,:))
    predictedCO2Data(i+1,:) = outCO2(i,:);
    hold on
end
save("CO2.mat" , 'predictedCO2Data');
legend(titleName(1),titleName(2),titleName(3),titleName(4),titleName(5));
title("expected CO2 value of different Countries")


figure(3)
predictedTempData = zeros(6,29);
predictedTempData(1,:) = 1:1:29;
for i = 1:5
    plot(outTemp(i,:))
    predictedTempData(i+1,:) = outTemp(i,:);
    hold on
end
save("temp.mat" , 'predictedTempData');
legend(titleName(1),titleName(2),titleName(3),titleName(4),titleName(5));
title("average temperture of different Countries")


aDiffData = zeros(2,yearLength);
aDiffData(1,:) = outCO2(1,:);
aDiffData(2,:) = outTemp(1,:);
filename = append('a',num2str(a),'.mat');
save(filename, 'aDiffData');

%% Function Definition

function [R2,MAE,RMSE,MAPE,error,PRD] = solve(p,q)
    error = q-p;
    R2 = 1 -( sum(error.^2) / sum((q-mean(p)).^2)  );
    RMSE = sqrt(mean(error.^2));
    MAE = mean(abs(error));
    MAPE = mean(abs(error)/p);
    PRD = 1/sqrt(1-R2^2);
end