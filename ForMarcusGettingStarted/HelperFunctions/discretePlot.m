function discretePlot(jawStrings, toothGapCenterIndexList, selectedRow, varargin)
    if nargin == 5
        targetFig = varargin{1};
        position = varargin{2};
        hold(targetFig, 'on');
    elseif nargin == 4
        targetFig = varargin{1};
        position = 0;
        hold(targetFig, 'on');
    elseif nargin == 3
        %figure;
        position = 0;
        hold on;
    end
    
    toothSelectColour = [1.00,0.87,0.20];
    gapSelectColour = [0.70,0.35,0.00];
    
    
    squareWidth = 1;
    toothGrid = [];
    gapGrid = [];

    for i = 1:length(jawStrings)
        toothArray = [];
        gapArray = [];
        leftIndex = toothGapCenterIndexList(i) - 1;
        rightIndex = toothGapCenterIndexList(i);
        currentCenter = toothGapCenterIndexList(i);
        currentJaw = jawStrings{i};
        
        %rightDist = 0;
        while rightIndex <= length(currentJaw)
            if currentJaw(rightIndex) == 'T'
                if rightIndex == position && i == selectedRow
                    selected = [rightIndex-currentCenter i];
                    positionColour = toothSelectColour;
                else
                    toothArray = [toothArray; squareWidth*(rightIndex-currentCenter)];
                end
            else
                if rightIndex == position && i == selectedRow
                    selected = [rightIndex-currentCenter i];
                    positionColour = gapSelectColour;
                else
                    gapArray = [gapArray; squareWidth*(rightIndex-currentCenter)];
                end
            end
            
            rightIndex = rightIndex + 1;
        end
        
        %leftDist = -1;
        while leftIndex > 0
            if currentJaw(leftIndex) == 'T'
                if leftIndex == position && i == selectedRow
                    selected = [leftIndex-currentCenter i];
                    positionColour = toothSelectColour;
                else
                    toothArray = [squareWidth*(leftIndex - currentCenter); toothArray];
                end
            else
                if leftIndex == position && i == selectedRow
                    selected = [leftIndex-currentCenter i];
                    positionColour = gapSelectColour;
                else
                    gapArray = [squareWidth*(leftIndex - currentCenter); gapArray];
                end
            end
            
            leftIndex = leftIndex - 1;
        end
        
        toothGrid = [toothGrid; toothArray ones(length(toothArray),1)*i];
        gapGrid = [gapGrid; gapArray ones(length(gapArray),1)*i];
    end

    toothSelect = toothGrid(toothGrid(:,2) == selectedRow, :);
    gapSelect = gapGrid(gapGrid(:,2) == selectedRow, :);
    
    if nargin > 3
        scatter(targetFig, toothGrid(:,1), toothGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0.9 0.1 0.1])
        scatter(targetFig, gapGrid(:,1), gapGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0 0 0])
        scatter(targetFig, toothSelect(:,1), toothSelect(:,2), 's', 'filled', 'MarkerFaceColor', [.15 0.9 0.2])
        scatter(targetFig, gapSelect(:,1), gapSelect(:,2), 's', 'filled', 'MarkerFaceColor', [0 0.3 0])
        if nargin == 5
            try
                scatter(targetFig, selected(1), selected(2), 's', 'filled', 'MarkerFaceColor', positionColour)
            catch
            end
        end
    elseif nargin == 3
        scatter(toothGrid(:,1), toothGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0.9 0.1 0.1])
        scatter(gapGrid(:,1), gapGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0 0 0])
        scatter(toothSelect(:,1), toothSelect(:,2), 's', 'filled', 'MarkerFaceColor', [.15 0.9 0.2])
        scatter(gapSelect(:,1), gapSelect(:,2), 's', 'filled', 'MarkerFaceColor', [0 0.3 0])
    end
    
end