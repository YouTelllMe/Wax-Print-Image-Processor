% Created by Roxanne

function [v0,Residuals] = BottomUpFit(xFull,yFull,v0,exponent, varargin)
    if nargin == 5
        showImage = varargin{1};
    else 
        showImage = true;
    end
    
    % Fit a conic section to the first N0 data points 
    N0=10;

    x=xFull(1:N0,:);
    y=yFull(1:N0,:);

    v0=FitData(x,y,v0,exponent);
    if showImage
        PlotFitAndData(x,y,xFull,yFull,v0)
    end

    % Fits a conic section to the rest of the data, adding one point at a time,
    % iteratively
    for k=1:length(xFull)-N0

        x=xFull(1:N0+k,:);
        y=yFull(1:N0+k,:);
        v0=FitData(x,y,v0,exponent);
        if showImage
            PlotFitAndData(x,y,xFull,yFull,v0)
        end
        pause(0.01)%why is there a pause here? if pause is necessary, can the time be shortened?
    end

    M=[x.^2 x.*y y.^2 x y];
    b=-ones(size(x));
    Residuals=(abs(M*v0-b).^exponent);%calculates residuals of final fit


    % figure(2)
    % hist(Residuals)

end


