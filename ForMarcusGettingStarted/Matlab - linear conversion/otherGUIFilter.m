
[mImg,nImg]=size(img);
[x,y]=meshgrid(1:nImg,1:mImg);
exp1=exp(-1.5);exp2=exp(-12);
mask1=1-exp(-exp1*((x-nImg/2).^2+(y-mImg/2).^2)); %    for high pass filtering
mask2=exp(-exp2*((x-nImg/2).^2+(y-mImg/2).^2));   %    for low pass filtering


figure
fig1=gcf;
colormap(gray)
set(fig1,'menubar','none','position',[5 288 512 384],'name',...
               'High and Low pass filting','numbertitle','off')
clf
subplot('position',[0.2 0 0.8 1])
h_mask=imagesc(mask1.*mask2);
axis off

 
maskMod=1;

 
figure(fig1)

 
cllbk1=['exp1=exp(get(h_highPass,''value''));mask1=1-exp(-exp1*((x-nImg/2).^2+(y-mImg/2).^2));'...
    'delete(h_mask);h_mask=imagesc(mask1.*mask2);axis off;maskMod=1;'...
    'set(h_apply,''backgroundcolor'',''red'');'...
    'set(h_text1,''string'',sprintf(''%02.2f'',get(h_highPass,''value'')))'];
cllbk2=['exp2=exp(get(h_lowPass,''value''));mask2=exp(-exp2*((x-nImg/2).^2+(y-mImg/2).^2));'...
    'delete(h_mask);h_mask=imagesc(mask1.*mask2);axis off;maskMod=1;'...
    'set(h_apply,''backgroundcolor'',''red'');'...
    'set(h_text2,''string'',sprintf(''%02.2f'',get(h_lowPass,''value'')))'];

 
cllbkApp=['figure(fig2);delete(h_im);'...
    'imgFilt=real(ifft2(ifftshift( fftshift(fft2(img)).*mask1.*mask2 )));'...
    'h_im=imagesc(imgFilt);axis off;maskMod=0;figure(fig1);'...-
    'set(h_apply,''backgroundcolor'',[0.7 0.7 0.7])'];

 
h_highPass=uicontrol('style','slider','position',[20 90 25 280],'callback',cllbk1,...
         'min',-20,'max',-0.1,'sliderstep',[0.01 0.1],'value',log(exp1));
h_lowPass=uicontrol('style','slider','position',[60 90 25 280],'callback',cllbk2,...
         'min',-20,'max',-0.1,'sliderstep',[0.01 0.1],'value',log(exp2));

 
h_text1=uicontrol('style','text','string',get(h_highPass,'value'),'position',[5 70 45 15]);
h_text2=uicontrol('style','text','string',get(h_lowPass,'value'),'position',[55 70 45 15]);

 
h_apply=uicontrol('style','pushbutton','callback',cllbkApp,...
              'position',[20 40 55 20],'string','Apply','backgroundcolor','red');
h_done=uicontrol('style','pushbutton','callback','if ~maskMod; done=1; end',...
             'position',[20 15 55 20],'string','Done');

 
figure
fig2=gcf;
colormap(gray)
set(fig2,'menubar','none','position',[520 288 512 384],'name',...
               'Filtered image','numbertitle','off')
clf
h_im=imagesc(img);
pause(0.5)
eval(cllbkApp);
figure(fig2)
h_im=imagesc(imgFilt);
axis off

         
done=0;
while ~done
    pause(0.1)
end

 
delete([fig1 fig2])