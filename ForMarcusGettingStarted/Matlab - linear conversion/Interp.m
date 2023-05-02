
figure(4)
colormap gray
imagesc(imgFilt);
hold on;

xplot=linspace(200,3000);
yplot=(-(B*xplot+E)-sqrt((B*xplot+E).^2-4*C*(A*xplot.^2+D*xplot+1)))/(2*C);
plot(xplot,yplot,'LineWidth',5);
vnew=zeros(25,128);
syms f(x);
f(x)=(-(E+B*x)-sqrt((E+B*x).^2-4*C*(A*x.^2+D*x+1)))/(2*C);
df=diff(f,x);
xpoint1=256.7;
dfpoint1=df(xpoint1);
gradient=double(dfpoint1);
yintcp=poly(xpoint1)+xpoint1/gradient;
normpoly=@(x) -x./gradient+yintcp;
% xq=linspace(xpoint1-100,xpoint1+100,25);
% yq=normpoly(xq);
xq=linspace(xpoint1-20/sqrt(1+gradient^2),xpoint1+20/sqrt(1+gradient^2),25);
yq=normpoly(xq);
v=[max(xq)-min(xq),normpoly(max(xq))-normpoly(min(xq))];
v=v/norm(v);
xq=linspace(xpoint1-70*abs(v(1)),xpoint1+70*abs(v(1)),25);
yq=normpoly(xq);
% [Xq,Yq]=meshgrid(xq,yq);
% idx =[1:25];
vnew(:,1)=interp2(im2double(imgFilt),xq,yq);
plot(xq,yq,'o',linspace(xpoint1-50*abs(v(1)),xpoint1+50*abs(v(1)),25),vnew(:,1),':.'); 

for i=1:769
    
    xpoint=xpoint1+5/sqrt(1+gradient^2);
    ypoint=poly(xpoint);
    dfpoint=df(xpoint);
    gradient=double(dfpoint);
    yintcp=poly(xpoint)+xpoint/gradient;
    normpoly=@(x) -x./gradient+yintcp;
    xq=linspace(xpoint-20/sqrt(1+gradient^2),xpoint+20/sqrt(1+gradient^2),25);
    yq=normpoly(xq);
    v=[max(xq)-min(xq),normpoly(max(xq))-normpoly(min(xq))];
    v=v/norm(v);
    xq=linspace(xpoint-70*abs(v(1)),xpoint+70*abs(v(1)),25);
    yq=normpoly(xq);
%     [Xq,Yq]=meshgrid(xq,yq);
%     idx=[25*i:25*i+24];
    vnew(:,i+1)=interp2(im2double(imgFilt),xq,yq);
    plot(xq,yq,'o',linspace(xpoint-50*abs(v(1)),xpoint+50*abs(v(1)),25),vnew(:,i+1),':.'); 
    xpoint1=xpoint;
      
end

figure(5) 
colormap gray
%imgnew=im2uint8(vnew);
%straigthened=imagesc(imgnew);
imagesc(vnew)
StraightJawImg=vnew;
