function alignedArray = discreteAlign(unalignedTeeth)
    array = repmat('-',length(unalignedTeeth), 150);
    for i = 1:length(unalignedTeeth)
        array(i,1:length(char(unalignedTeeth(i)))) = unalignedTeeth(i);
    end
    array = [repmat('-', length(unalignedTeeth), 75) array];


    dataStart = 76;
    dataEnd = length(char(unalignedTeeth(1))) + dataStart - 1;
    dataMin = dataStart;
    dataMax = dataEnd;
    for i = 1 : length(unalignedTeeth)-1
        start = localalign(unalignedTeeth(i), unalignedTeeth(i+1), 'Alphabet', 'NT', 'GapOpen', 75).Start;
        %nwalign(unalignedTeeth(i), unalignedTeeth(i+1), 'Alphabet', 'NT', 'GapOpen', 75)
        diff = start(1) - start(2);
        %diff
        if diff > 0
            array(i+1:end, :) = [repmat('-', length(unalignedTeeth)-i, diff) array(i+1:end, 1:end-diff)];
        elseif diff < 0
            array(i+1:end, :) = [array(i+1:end, 1+abs(diff):end) repmat('-', length(unalignedTeeth)-i, abs(diff))];
        else%diff == 0
            %no shift required
        end
        dataStart = dataStart + diff;
        dataEnd = dataStart + length(char(unalignedTeeth(i+1))) - 1;
        if dataStart < dataMin
            dataMin = dataStart;
        end
        if dataEnd > dataMax
            dataMax = dataEnd;
        end
    end


    alignedArray = array(:,dataMin:dataMax);
end