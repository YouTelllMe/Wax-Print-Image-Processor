

%Consider removing centerToothGapIndex, since we can calculate it more
%accurately in centerJaw
function [charArray, lengthArray, gapLengths, centerToothGapIndex] = convertLengths(lengthArray)
    toothMaxWidth = 70;
    avgToothWidth = 45;
    gapLengths = [];
    charArray = repmat('T', 1, length(lengthArray));

    i = 1;
    while i < length(lengthArray)-1
        %finds gaps to interpolate data for
        interToothDist = lengthArray(i+1)-lengthArray(i);
        if interToothDist > toothMaxWidth
            %gapsToAdd = round((interToothDist)/avgToothWidth);%not sure how well this will deal with overlapping distributions
            gapsToAdd = 1;
            gapDist = interToothDist/(gapsToAdd + 1);
        elseif interToothDist < toothMaxWidth
            gapsToAdd = 0;
        end
        
        %adds gap data via interpolation
        while gapsToAdd > 0
            lengthArray = [lengthArray(1:i); lengthArray(i) + gapDist; lengthArray(i+1:end)];
            charArray = [charArray(1:i) 'G' charArray(i+1:end)];
            gapLengths = [gapLengths ; lengthArray(i) + gapDist];
            i = i+1;
            gapsToAdd = gapsToAdd - 1;
        end
        i = i+1;
    end
    
    [~, centerToothGapIndex] = min(abs(lengthArray));
    
end