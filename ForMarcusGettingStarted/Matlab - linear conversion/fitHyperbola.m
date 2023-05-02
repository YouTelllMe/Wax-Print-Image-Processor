%This function takes an image and an array of points on that image and fits
%a hyperbola to the points in the array. Format of point array is an m x 2
%array with x coordinate in column 1 and y coordinate in column 2, one pair
%per row.
%
%Created by Roxanne, edited by Alex Fraser


function [pointsFitted, v0] = fitHyperbola(img, points, varargin)
    if nargin == 4
        showImage = varargin{1};
        cullPoints = varargin{2};
    elseif nargin == 3
        showImage = varargin{1};
        cullPoints = true;
    else 
        showImage = true;
        cullPoints = true;
    end
    
    xFull = points(:,1);
    yFull = points(:,2);

    t=[-1:0.01:1.2]';
    x0=8*sec(t)+0.01*randn(size(t));
    y0=3*tan(t)+0.01*randn(size(t));

    % Rotate all the hyberbola
    % theta=rand;
    % rotation=[cos(pi*theta) sin(pi*theta); -sin(pi*theta) cos(pi*theta)];
    % new=[x0,y0]*rotation;
    % x=new(:,1);
    % y=new(:,2);

    % The "data" - the hyberbola above plus some random points thrown in
    %xFull=cent(1:2:end)';
    %yFull=cent(2:2:end)';

    % Rearrange the points randomly to test the robustness of the iterative
    % algorithm
    % NewOrder=randperm(length(xFull));
    % xFull=xFull(NewOrder)';
    % yFull=yFull(NewOrder)';

    exponent=0.7;
    %v0 = [1 0 -1 0 0]'; %defines initial hyperbola
    v0 = [6.60438211815553e-07;-1.25187320702260e-07;-2.15418071984042e-07;-0.00155917769024401;-1.31862413938453e-05];%defines initial hyperbola, which should be close to jaw already
    
    [v0,Residuals]=BottomUpFit(xFull,yFull,v0,exponent, showImage);

    % Removes the biggest residual when above a threshold value, refitting after each
    % removal until no residuals exceed threshold
    if cullPoints
        while max(Residuals) > 0.1%default 0.1
            [MaxR indR]=max(Residuals);
            count=1:length(xFull);
            xFull(indR)=[];
            yFull(indR)=[];
            [v0,Residuals]=BottomUpFit(xFull,yFull,v0,exponent, showImage);
        end
    end
    
    
    %set hyperbola fit points to return
    pointsFitted(:,1) = xFull;
    pointsFitted(:,2) = yFull;

    coefficients=FitData(xFull,yFull,v0,2);
    %PlotFitAndData(xFull,yFull,xFull,yFull,coefficients);
    %axis([min(xFull) max(xFull)  min(yFull)-100 max(yFull)])
    %      figure(2)
    %      hist(Residuals)

    
    %why is this huge code block here? it seems to be finding the equation
    %of the fit again using a standard parabola instead of hyperbola?
    
    %{
    %following section is identical to myFit(). why is it here?
    xy=[xFull(:),yFull(:)].';                           
    theta= fminsearch(@(theta) cost(theta,xy), 45);     %minimizes an angle, and stores angle in theta variable?
    [~,coeffs]=cost(theta,xy);                          %
    [a,b,c]=deal(coeffs(1),coeffs(2), coeffs(3));       %
    xv=-b/2/a;                                          %
    vertex=rotateMatDeg(-theta)*[xv;polyval(coeffs,xv)];%is this the same vertex as the hyperbola?
    
    Rxy=rotateMatDeg(theta)*xy;     %
    [xx,idx]=sort(Rxy(1,:));        %
    yy=Rxy(2,idx);                  %this block is the same as the cost function, why is it here?
    [coeffs,S]=polyfit(xx,yy,2);    %
    Cost=S.normr;                   %
    
    Rmat=[cosd(theta), -sind(theta); sind(theta), cosd(theta)];% rotation matrix, same as rotateMatDeg function, why is it here?
    
    xFullsort=sort(xFull);
    arclen=zeros(size(xFull));
    A=v0(1);B=v0(2);C=v0(3);D=v0(4);E=v0(5);%coefficients of hyperbola
    poly=@(x) (-(E+B*x)-sqrt((E+B*x).^2-4*C*(A*x.^2+D*x+1)))/(2*C);%quadratic root equation?, figure out what this equation is!!!
    fitpoly=poly(xFullsort);
    for i=1:size(xFull)-1
        arclen(i)=arclength([xFullsort(i),xFullsort(i+1)],[fitpoly(i),fitpoly(i+1)],'s');
    end

    x=zeros(length(arclen)+1,1);
    x(1)=xFullsort(1);
    for i=1:size(arclen)
        x(i+1)=x(i)+arclen(i);
    end

    %}
    
    
    if showImage
        %plot fitted points on the original image
        imshow(img);
        %imshow (line 337) %Why doesn't this line work? what is line 337?
        hold on
        %plot(xFullsort,poly(xFullsort),'.-','Markersize',25) % plots the remaining points, y coord based on hyperbola function?
        PlotFitAndData(xFull, yFull, xFull, yFull, v0, false)
    end

    %plot(x,ones(size(x)) * 300,'o-','LineWidth',5,'MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',2)
    %hist(diff(x),20)

    %vertex=vertex1;
    % vertex=vertex2;
    % d=vertex1(1)-vertex2(2);
    % x1new=x1-d;
    %hold on
    %plot(x1,ones(size(x1)) * 300+0.01,'o-','LineWidth',5,'MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',2);

end
