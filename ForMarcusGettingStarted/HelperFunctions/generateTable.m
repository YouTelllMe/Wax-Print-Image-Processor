function [fullJaw, mirrorJaw] = generateTable(filenames, jawStrings, centerIndex)
    %generateCSV Summary of this function goes here
    %   Detailed explanation goes here
    if ~(length(filenames) == length(jawStrings) && length(jawStrings) == length(centerIndex))
        ME = MException("Argument arrays must have same length!");
        throw(ME)
    end

    for i = 1:length(jawStrings)
        jawStrings{i} = [repmat(' ',1,51-centerIndex(i)) jawStrings{i}];
        jawStrings{i} = [jawStrings{i} repmat(' ',1,101-length(jawStrings{i}))];
        filenames{i} = remFolderStruct(filenames{i});
        jawStrings{i} = replace(jawStrings{i}, "G", "0");
        jawStrings{i} = replace(jawStrings{i}, "T", "1");
    end
    data = char(jawStrings);,

    positionNames = string(num2str([-50:1:50]'));
    dataTable = array2table(data, 'VariableNames', positionNames', 'RowNames', filenames);

    fullJaw = dataTable;
    mirrorJaw = '';
end