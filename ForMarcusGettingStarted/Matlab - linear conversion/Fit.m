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
xFull=cent(1:2:end)';
yFull=cent(2:2:end)';

% Rearrange the points randomly to test the robustness of the iterative
% algorithm
% NewOrder=randperm(length(xFull));
% xFull=xFull(NewOrder)';
% yFull=yFull(NewOrder)';

exponent=0.7;
v0=[1 0 -1 0 0]';

[v0,Residuals]=BottomUpFit(xFull,yFull,v0,exponent);

while max(Residuals)>0.1
    [MaxR indR]=max(Residuals);
    count=1:length(xFull);
    xFull(indR)=[];
    yFull(indR)=[];
    [v0,Residuals]=BottomUpFit(xFull,yFull,v0,exponent);
        
end

%remove big residuals and then refit
%threshold of residual improvement
    
coefficients=FitData(xFull,yFull,v0,2);
%PlotFitAndData(xFull,yFull,xFull,yFull,coefficients);
%axis([min(xFull) max(xFull)  min(yFull)-100 max(yFull)])
%      figure(2)
%      hist(Residuals)

xy=[xFull(:),yFull(:)].';
theta= fminsearch(@(theta) cost(theta,xy), 45);    
[~,coeffs]=cost(theta,xy);
[a,b,c]=deal(coeffs(1),coeffs(2), coeffs(3));
xv=-b/2/a;
vertex=R(-theta)*[xv;polyval(coeffs,xv)];
Rxy=R(theta)*xy;
[xx,idx]=sort(Rxy(1,:));
yy=Rxy(2,idx);
[coeffs,S]=polyfit(xx,yy,2);
Cost=S.normr;
Rmat=[cosd(theta), -sind(theta); sind(theta), cosd(theta)];
xFullsort=sort(xFull);
arclen=zeros(size(xFull));
A=v0(1);B=v0(2);C=v0(3);D=v0(4);E=v0(5);
poly=@(x) (-(E+B*x)-sqrt((E+B*x).^2-4*C*(A*x.^2+D*x+1)))/(2*C);
fitpoly=poly(xFullsort);
for i=1:size(xFull)-1
    arclen(i)=arclength([xFullsort(i),xFullsort(i+1)],[fitpoly(i),fitpoly(i+1)],'s');
end
 
x=zeros(length(arclen)+1,1);
x(1)=xFullsort(1);
for i=1:size(arclen)
    x(i+1)=x(i)+arclen(i);
end

%plot fitted points on the original image
% imshow(img);
% imshow (line 337) 
% hold on
% plot(xFullsort,poly(xFullsort),'.-','Markersize',25)


%plot(x,ones(size(x)) * 300,'o-','LineWidth',5,'MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',2)
%hist(diff(x),20)

%vertex=vertex1;
% vertex=vertex2;
% d=vertex1(1)-vertex2(2);
% x1new=x1-d;
%hold on
%plot(x1,ones(size(x1)) * 300+0.01,'o-','LineWidth',5,'MarkerFaceColor','red','MarkerEdgeColor','red','MarkerSize',2);
