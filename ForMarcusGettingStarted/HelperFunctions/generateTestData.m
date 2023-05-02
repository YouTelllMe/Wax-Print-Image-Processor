function [testData, scrambleData ] = generateTestData()
    testData = mod([0:30]', 11) + 1;
    for i = 1:80
        testData = [testData mod((testData(:,end) + 5), 11) + 1];
    end
    fun = @(x) (x > 9) * 'G' + (x <= 9) * 'T';
    testData = string(char(arrayfun(fun, testData)));
    scrambleData = testData;
    for i = 1:length(scrambleData)
        scrambleData(i) = extractBetween(scrambleData(i), round(rand()*10) + 1, length(scrambleData{i}) - round(rand()*10));
    end
end