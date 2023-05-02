function centeredArray = discreteCenter(uncenteredTeeth, centerIndexList)
    array = repmat('-',length(uncenteredTeeth), 150);
    for i = 1:length(uncenteredTeeth)
        array(i,1:length(char(uncenteredTeeth(i)))) = uncenteredTeeth(i);
    end
    array = [repmat('-', length(uncenteredTeeth), 75) array];


    dataStart = 76;
    dataEnd = length(char(uncenteredTeeth(1))) + dataStart - 1;
    dataMin = dataStart;
    dataMax = dataEnd;
    for i = 1 : length(uncenteredTeeth)-1
        
        diff = centerIndexList(i) - centerIndexList(i+1);
        %diff
        if diff > 0
            array(i+1:end, :) = [repmat('-', length(uncenteredTeeth)-i, diff) array(i+1:end, 1:end-diff)];
        elseif diff < 0
            array(i+1:end, :) = [array(i+1:end, 1+abs(diff):end) repmat('-', length(uncenteredTeeth)-i, abs(diff))];
        else%diff == 0
            %no shift required
        end
        dataStart = dataStart + diff;
        dataEnd = dataStart + length(char(uncenteredTeeth(i+1))) - 1;
        if dataStart < dataMin
            dataMin = dataStart;
        end
        if dataEnd > dataMax
            dataMax = dataEnd;
        end
    end

    centeredArray = array(:,dataMin:dataMax);
end