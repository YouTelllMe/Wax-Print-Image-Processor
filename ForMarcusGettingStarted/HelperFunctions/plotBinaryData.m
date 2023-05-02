function plotBinaryData(charArray)
    boxHeight = 80;
    boxWidth = 40;
    toothLoc = [];
    gapLoc = [];
    %graph = figure
    for i = 1:size(charArray, 1)
        for j = 1:size(charArray, 2)
            if charArray(i,j) == 'T'
                toothLoc = [toothLoc; j*boxHeight, i];
                %rectangle('Position', [j*boxWidth + 3*j, i*boxHeight + 3*i,boxWidth, boxHeight], 'FaceColor', [0 0 0])
            elseif charArray(i,j) == 'G'
                gapLoc = [gapLoc; j*boxHeight, i];
                %rectangle('Position', [j*boxWidth + 3*j, i*boxHeight + 3*i,boxWidth, boxHeight])
            else
            end
        end
    end
    figure
    hold on
    scatter(toothLoc(:,1), toothLoc(:,2), boxHeight, 's', 'filled', 'MarkerFaceColor', [0 0 0] );
    scatter(gapLoc(:,1), gapLoc(:,2), boxHeight, 's', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', [0.85 0.85 0.85]);
    hold off

end