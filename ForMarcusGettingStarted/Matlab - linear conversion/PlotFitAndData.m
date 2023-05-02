%Plots datapoints and the conic fit of those data points

function PlotFitAndData(x,y,xFull,yFull,v0, varargin)

    clearImage = true;
    if nargin == 6
        clearImage = varargin{1};
    end

    %xx1 = [-max(xFull)-100 : (max(xFull)-min(xFull))/100 : max(xFull)];% these lines fail when filling from negative to postitve
    %yy1 = [-max(yFull)-100 : (max(yFull)-min(yFull))/100 : max(yFull)];% potential fix: use 'min(xFull) - 100' instead of '-max(xFull) - 100'

    xx1 = [min(xFull)-100 : (max(xFull)-min(xFull))/100 : max(xFull)+100]; 
    yy1 = [min(yFull)-100 : (max(yFull)-min(yFull))/100 : max(yFull)+100];
    
    [xx, yy] = meshgrid(xx1,yy1);

    %creates equation to calculate height of a surface
    zz=v0(1)*xx.^2 + v0(2)*xx.*yy + v0(3)*yy.^2 + v0(4)*xx + v0(5)*yy + 1; %how did we determine 1 as the constant? % because any potential solution can be scaled to have constant = 1 and be the same curve

    %figure(1)
    if clearImage
        clf
    end
    axis equal
    hold on
    
    %this is the contour plot of z = 0, which is our conic section
    contour(xx,yy,zz,[0 0],'linewidth',2,'color','g')
    plot(xFull,yFull,'b*')
    plot(x,y,'r*')
    hold off

end