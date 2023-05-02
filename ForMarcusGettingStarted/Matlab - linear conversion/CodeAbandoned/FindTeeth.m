
%StraightJawImg=imread('Sample.png');
%StraightJawImg=StraightJawImg(:,:,1);

figure(5)
colormap gray
clf
hold on

M=size(StraightJawImg,1);
N=size(StraightJawImg,2);

imagesc(StraightJawImg)
coords=ginput(2);
x1=coords(1,1);
x2=coords(2,1);
y1=coords(1,2);
y2=coords(2,2);
hold on
plot([x1 x1 x2 x2 x1],[y1 y2 y2 y1 y1])
xlow=round(min(x1,x2));
xhigh=round(max(x1,x2));
ylow=round(min(y1,y2));
yhigh=round(max(y1,y2));

%%
count=0;

while xhigh < N & count<20

    count=count+1;
    
    SearchFor=StraightJawImg(ylow:yhigh,xlow:xhigh);
    SearchIn=StraightJawImg(:,max(round(xhigh-(xhigh-xlow)/5),1):min(round(xhigh+6*(xhigh-xlow)/5),size(StraightJawImg,2)));

    figure(6)
    colormap gray
    clf
    hold on
    imagesc(SearchFor)

    figure(7)
    colormap gray
    clf
    hold on
    imagesc(SearchIn)

    SearchIn=SearchIn-mean(mean(SearchIn));
    xcor=normxcorr2(SearchFor,SearchIn);
    [xpeak, ypeak] = find(xcor==max(xcor(:)));
    match(count)=max(xcor(:));

    xcorSelf=normxcorr2(SearchFor,SearchFor);
    [xpeakSelf, ypeakSelf] = find(xcorSelf==max(xcorSelf(:)));

    ylow = max(ypeak - ypeakSelf + 1,1);
    yhigh = min(ylow + size(SearchFor,2),M);
    xlow = max(xpeak - xpeakSelf + 1,1);
    xhigh = min(xlow + size(SearchFor,1),N);


    im=imagesc(ylow:yhigh,xlow:xhigh,SearchFor);
    im.AlphaData = 0.2;    % set transparency

    xlow = xlow + max(round(xhigh-(xhigh-xlow)/5),1);
    xhigh = xhigh + max(round(xhigh-(xhigh-xlow)/5),1);
    
    figure(5)
    plot([xlow xlow xhigh xhigh xlow],[ylow yhigh yhigh ylow ylow])

end


