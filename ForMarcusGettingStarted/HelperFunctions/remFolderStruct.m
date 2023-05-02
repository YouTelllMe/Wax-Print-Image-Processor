function outputString = remFolderStruct(inputString)
    if ismac
        % Linux code might work on mac too, not sure.
        disp('Mac may not be supported, check remFolderStruct() code')
    elseif isunix
        outputString = regexprep(inputString, '/((\w*\s?)*/)*', '');
    elseif ispc
        outputString = regexprep(inputString, 'C:\\((\w*\s?)*\\)*', '');
    else
        disp('Platform not supported')
    end
end