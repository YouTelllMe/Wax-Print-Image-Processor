%rawData = load('TrainingSet/export21images_LG244Fixed2.mat');
rawData = load('LG 281/exportLabels_LG281_2020Spring.mat');
cleanData = objectDetectorTrainingData(rawData.gTruth);
sortedData = sortrows(cleanData, 1);%sorts image files by date via name
imageList =  sortedData{:,1};

%this loop brings all teeth values into an array of xyz locations
xyzCoords = [];
xyzCoordsShifted = [];
%clf;
%hold on;
%thetaComparison = [];%changed rotation method, these lines not needed
%axisComparison = [];
centers = [];
solns = [];
arcGrid = [];
lengthGrid = [];
gapGrid = [];
alignGrid = [];
alignGapGrid = [];
centerToothGrid = [];
centerGapGrid = [];
alignGapToothGrid = [];
alignGapGapGrid = [];
distList = [];
distListAlt = [];
centerIndexList = [];
toothGapCenterIndexList = [];

xyToothTable = table('Size', [height(sortedData) 1], 'VariableTypes', "cell");
xyGapTable = table('Size', [height(sortedData) 1], 'VariableTypes', "cell");

for i = 1:height(sortedData)
    tic
    %load image
    imgName = sortedData{i,1};
    imgName = imgName{1,1};
    img = rgb2gray(imread(imgName));
    %load, sort, and format data
    xyData = sortedData{i,2};
    xyData = xyData{1,1};
    xyCoords = xyData(:,1:2) + xyData(:,3:4) / 2;
    pause(1);
    xyCoords = sortrows(xyCoords, 1);
    xyCoords = sortrows(xyCoords, 1);
    %fit hyperbola and cull bad points
    [fitPoints, coeff] = fitHyperbola(img, xyCoords, false, false); %something is wrong here, good points are being culled improperly!!!
    %Bad culls are usually not a problem, but occasionally I get an
    %equation with no real solutions for the vertex. This causes problems
    %downstream in calculation of vertex to align the jaws.
    
    %find 'matrix of the quadratic equation'
    matQuadFull = [coeff(1) coeff(2)/2 coeff(4)/2; coeff(2)/2 coeff(3) coeff(5)/2; coeff(4)/2 coeff(5)/2 1];
    %find 'matrix of the quadratic form'
    matQuad = [coeff(1) coeff(2)/2; coeff(2)/2 coeff(3)];
    %find eigenvectors of the 'matrix of the quadratic form', which are
    %parallel to major and minor axes of hyperbola or ellipse
    [eVectors, eValues] = eig(matQuad);

    %Optional visualization at fitting step
    %PlotFitAndData(fitPoints(:,1), fitPoints(:,2), fitPoints(:,1), fitPoints(:,2), coeff, true);

    
    %implement code HERE to make sure eigenvectors are in correct order?
    
    
    %{
    alternate way of rotating points to line up. Abandoned because it was
    not clear how to align vertices.
    refVector = [0; -1];
    majorAxis = eVectors(:,1); %consider choosing axes properly, don't think the first vector is guaranteed major instead of minor axis
    minorAxis = eVectors(:,2);
    theta = angleToVector(refVector, majorAxis);
    newAxis = rotateMat(theta) * majorAxis;
    rotPoints = (rotateMat(theta) * fitPoints')';
    %}
    
    rotPoints = (eVectors * xyCoords')';%eigenvectors make a rotation matrix to normalize data to [0; 0] reference vector
    BVec = [coeff(4) coeff(5)];
    rotCoeff = [eValues(1,1) 0 eValues(2,2) BVec*eVectors(:,1) BVec*eVectors(:,2)];%graphing fixed, lower bound error in PlotFitAndData fixed
    majorAxis = [-1; 0];%After rotation, this is the major axis vector
    
    %PlotFitAndData(rotPoints(:,1), rotPoints(:,2), rotPoints(:,1), rotPoints(:,2), rotCoeff, true);
    
    %cannot use centers to align, because they move too far along major axis in response to
    %hyperbola curvature, and are in totally different locations in
    %hyperbola vs ellipse
    %x0 = -BVec*eVectors(:,1)/2;%this is supposed to calculate center, but it gives wrong answer?
    %y0 = -BVec*eVectors(:,2)/2;

    A=rotCoeff(1); B=rotCoeff(2); C=rotCoeff(3); D=rotCoeff(4); E=rotCoeff(5);%coefficients of rotated hyperbola
    %align hyperbola center to initialCenter
    %center = [(B*E -
    %2*C*D)/(4*A*C-B*B),(D*B-2*A*E)/(4*A*C-B*B)];%alternate method of finding center, answer should be identical to below
    centerMatrix = rref([2*A B -D; B 2*C -E]);%this solves the partial derivatives of Qx=0 and Qy=0 (Q is cartesian conic matrix)
    center = centerMatrix(:,3)';%NOTE: centers are very far apart in x direction, cannot be used for alignment, we find and use vertex instead
    centers = [centers; center];
    
    %Solve quadratic to find hyperbola vertices
    yv = center(2);
    %A*x*x + B*x*yv + C*yv*yv + D*x + E*yv + 1 == 0; %simplifies to quadratic below
    a = A; b = B*yv + D; c = C*yv*yv + E*yv + 1;
    soln = [(-b + sqrt(b*b - 4*a*c))/(2*a) (-b - sqrt(b*b - 4*a*c))/(2*a)];
    if imag(soln) ~= 0
        error("No real solutions to vertex position!")
    end
    solns = [solns ; soln];
    %choose correct vertex, because center is on different sides of jaw in
    %ellipse vs hyperbola
    if center(1) >= 0
        xv = min(soln);
    elseif center(1) < 0
        xv = max(soln);
    end
    %vertex is [xv,yv]
    
    %calculate vertex aligned points and equation
    shiftPoints = rotPoints - [xv yv];
    denom = A*xv*xv + C*yv*yv + D*xv + E*yv + 1;
    shiftCoeff = [A/denom 0 C/denom (D+2*A*xv)/denom (E+2*C*yv)/denom];
    
    %calculate centered points and equation
    xc = center(1);
    yc = center(2);
    centerPoints = rotPoints - center;
    centeredVertex = [xv-center(1) yv-center(2)];
    denom = A*xc*xc + C*yc*yc + D*xc + E*yc + 1;
    centerCoeff = [A/denom 0 C/denom (D+2*A*xc)/denom (E+2*C*yc)/denom];
    
    %PlotFitAndData(centerPoints(:,1), centerPoints(:,2), centerPoints(:,1), centerPoints(:,2), centerCoeff, true);
    
    %calculate and plot centered equation in standard form, not aligned on vertices
    %because of center location varying based on asymptote angle in
    %hyperbola and different sign in ellipse centers
    
    K = -det(matQuad) / det(matQuadFull);
    S = det(matQuadFull);
    aa = -S / (eValues(1,1).^2 * eValues(2,2));%a^2 term
    bb = -S / (eValues(2,2).^2 * eValues(1,1));%b^2 term
    
    %{
    %plot centered equation
    hold on
    %xHyp = [min(shiftPoints(:,1))*20 : 0.1 : max(shiftPoints(:,1))*20];
    xHyp = [-20000 : 0.1 : 20000];
    %plot(xHyp, sqrt(bb*(1 - xHyp.^2/aa)), 'k', xHyp, -sqrt(bb*(1 - xHyp.^2/aa)), 'k')%plots x^2/a^2 + y^2/b^2 = 1. Works for both ellipses and hyperbolas.
    plot(xHyp, sqrt(bb*(1 - xHyp.^2/aa)), 'k')%plots x^2/a^2 + y^2/b^2 = 1. Works for both ellipses and hyperbolas.
    plot(xHyp, -sqrt(bb*(1 - xHyp.^2/aa)), 'k')
    plot(centeredVertex(1), centeredVertex(2), 'b*')
    %}
    
        %this code block is used to arclength calculations
        %Arclength didn't seem to work well for the purposes of the tooth
        %data.
    %{
    %Calculate arclengths
    arcLengths = [];
    for k = 1:length(centerPoints)
        arcLength = findArcLength(centerPoints(k,1), centerPoints(k,2), centeredVertex(1), centeredVertex(1), aa, bb);
        arcLengths = [arcLengths; arcLength];
    end
    
    arcGrid = [arcGrid; arcLengths ones(length(arcLengths),1)*i];
    %}
    
    %Calculate nearest points on line
    linePoints = [];
    for j = 1:length(centerPoints)
        linePoint = findNearest(centerPoints(j,1), centerPoints(j,2), aa, bb);
        linePoints = [linePoints; linePoint];
    end
    
    %PlotFitAndData(centerPoints(:,1), centerPoints(:,2), linePoints(:,1), linePoints(:,2), centerCoeff, true);
    
    %Sort coordinates based on nearest line points. ***NOTE***: Other arrays may
    %not match correct tooth order unless they are also sorted at this step!
    %This step fixes some adjacent teeth that are in opposite orders in the
    %array, which was causing excess gaps to be interpolated.
    sortTemp = sortrows([linePoints xyCoords shiftPoints], 2, 'descend');
    linePoints = sortTemp(:,1:2);
    xyCoords = sortTemp(:,3:4);
    shiftPoints = sortTemp(:,5:6);
    
    [~, gapCoords] = interpolate2DGaps(xyCoords);
    [toothGapPoints, ~] = interpolate2DGaps(shiftPoints);
    
    %Convert xy coordinates of tooth positions into continuous 1D representation
    [pointLength, pointDist] = findPointLength(shiftPoints);
    
    %convert continuous length array to binary position array
    [charArray, toothGapLengthArray, gapLengths, ~] = convertLengths(pointLength);
    
    if i == 1
        charGrid = string(charArray);
    elseif i > 1
        charGrid = [charGrid; string(charArray)];
    end
    
    %align arrays based on locating best center tooth
    [centerToothGapIndex, linearShift] = centerJaw(toothGapLengthArray, toothGapPoints, majorAxis, img);
    %showToothPairs(img, toothGapPoints, centerToothGapIndex, majorAxis)
    adjustedCenterIndex = find( pointLength == toothGapLengthArray(centerToothGapIndex));
    toothGapCenterIndexList = [toothGapCenterIndexList; centerToothGapIndex];
    centerIndexList = [centerIndexList; adjustedCenterIndex];
    
    centeredJawTeeth = pointLength - linearShift;
    centeredJawGaps = gapLengths - linearShift;
    
    centerToothGrid = [centerToothGrid; centeredJawTeeth ones(length(pointLength),1)*i];
    centerGapGrid = [centerGapGrid; centeredJawGaps ones(length(centeredJawGaps),1)*i];
    
    %{
    %These functions aligned jaws based on the linear scoring matrix
    method, but it did not work well on either teeth or gap points that
    well. centerJaw works much better than either of these!
    
    %align jaw lengths to previous jaw lengths, then advance previous jaw
    %length variable
    if i > 1
        alignedJawTeeth = pointLength - align1DVectors(prevJaw, pointLength);
        alignedJawGaps = gapLengths - align1DVectors(prevJaw, pointLength);
    elseif i == 1
        alignedJawTeeth = pointLength;
        alignedJawGaps = gapLengths;
    end
    alignGrid = [alignGrid; alignedJawTeeth ones(length(pointLength),1)*i];
    alignGapGrid = [alignGapGrid; alignedJawGaps ones(length(alignedJawGaps),1)*i];
    prevJaw = alignedJawTeeth;
    
    %align jaw gaps to previous jaw gaps, then advance previous jaw
    %length variable
    if i > 1
        alignedGapGaps = gapLengths - align1DVectors(prevGaps, gapLengths);
        alignedGapTeeth = pointLength - align1DVectors(prevGaps, gapLengths);
        prevGaps = alignedGapGaps;
    elseif i == 1
        alignedGapGaps = gapLengths;
        alignedGapTeeth = pointLength;
        prevGaps = gapLengths;
    end
    alignGapToothGrid = [alignGapToothGrid; alignedGapTeeth ones(length(alignedGapTeeth),1)*i];
    alignGapGapGrid = [alignGapGapGrid; alignedGapGaps ones(length(alignedGapGaps),1)*i];
    %}
    
    

    %figure
    
    %plot(center(1), 0, 'b')
    %{
    figure
    PlotFitAndData(shiftPoints(centerIndexInitial,1), shiftPoints(centerIndexInitial,2), shiftPoints(:,1), shiftPoints(:,2), shiftCoeff, true);
    figure
    PlotFitAndData(shiftPoints(adjustedCenterIndex,1), shiftPoints(adjustedCenterIndex,2), shiftPoints(:,1), shiftPoints(:,2), shiftCoeff, true);
    %}
    
    %imshow(img)
    
    %PlotFitAndData(xv, yv, shiftPoints(:,1), shiftPoints(:,2), rotCoeff, false);
    %xline(center(1));
    %hold off
    %plot(xyCoords(:,1), xyCoords(:,2))
    %add points to full array
    distList = [distList; pointDist ones(length(pointDist),1)*i];
    lengthGrid = [lengthGrid; pointLength ones(length(pointLength),1)*i];
    gapGrid = [gapGrid; gapLengths ones(length(gapLengths),1)*i];
    xyToothTable{i,1} = {xyCoords};
    xyGapTable{i,1} = {gapCoords};
    xyzCoords = [xyzCoords; fitPoints ones(length(fitPoints),1)*i];
    xyzCoordsShifted = [xyzCoordsShifted; shiftPoints ones(length(shiftPoints),1)*i];
    
    disp("Elapsed time is " + toc + " seconds. Image " + i + "/" + height(sortedData))
end

%createJawArrayfromteethlocation -> laurens code

%shiftLengths = [];
%{
%arcgrid disabled for now
%shift lengthgrid to match arcgrid
for j = 1:i
    diff = [min(lengthGrid(lengthGrid(:,2) == j,1)) - min(arcGrid(arcGrid(:,2) == j,1)) 0];
    shiftLengths = [shiftLengths; lengthGrid(lengthGrid(:,2) == j,:) - diff];
end
%}

%clf;
%plot3(xyzCoords(:,1), xyzCoords(:,2), xyzCoords(:,3), '+');
%figure
%plot3(xyzCoordsShifted(:,1), xyzCoordsShifted(:,2), xyzCoordsShifted(:,3), '+');
%hold on
%arclengthdata = scatter(arcGrid(:,1), arcGrid(:,2), 'MarkerEdgeColor', [0.5 0 0])
%point2pointdata = scatter(shiftLengths(:,1), shiftLengths(:,2), '*', 'MarkerEdgeColor', [0 0.8 0.3])
figure
hold on
scatter(lengthGrid(:,1), lengthGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0.1 0.5 1])
scatter(gapGrid(:,1), gapGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0 0 0])
title("\fontsize{18}Unaligned jaws")
%{
%Outdated centering techniques
figure
hold on
scatter(alignGrid(:,1), alignGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0.9 0.1 0.1])
scatter(alignGapGrid(:,1), alignGapGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0 0 0])
title("\fontsize{22}Aligned jaws")
figure
hold on
scatter(alignGapToothGrid(:,1), alignGapToothGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0.9 0.1 0.1])
scatter(alignGapGapGrid(:,1), alignGapGapGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0 0 0])
title("\fontsize{22}Gap Aligned jaws")
%}
figure
hold on
scatter(centerToothGrid(:,1), centerToothGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0.9 0.1 0.1])
scatter(centerGapGrid(:,1), centerGapGrid(:,2), 's', 'filled', 'MarkerFaceColor', [0 0 0])
title("\fontsize{22}Center Aligned jaws")

figure
hold on
title("\fontsize{18}Discrete Center Aligned Jaws LG 281")
discretePlot(charGrid, toothGapCenterIndexList, 0)

figure
histogram(distList(:,1), 60)
title("\fontsize{18}Intertooth distances along jaw")
%figure
%histogram(distListAlt(:,1),60)
%flip(charGrid)

%discreteSeries = discreteAlign(charGrid)
%discreteCenterSeries = discreteCenter(charGrid, centerIndexList)