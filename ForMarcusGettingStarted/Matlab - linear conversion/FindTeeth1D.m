imagesc(StraightJawImg)

Jaw1D = max(StraightJawImg);

figure(5)
clf
title('Intensity along jaw (max projection), I(t)')
Jaw1Dsmoothed=movmean(movmean(Jaw1D,3),3);
plot(Jaw1Dsmoothed)



L=length(Jaw1Dsmoothed);
Y=fft(Jaw1Dsmoothed);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
% P1 = movmean(P1,3);
% P1 = movmean(P1,3);
% P1 = movmean(P1,3);
% P1 = movmean(P1,3);
% P1 = movmean(P1,3);
%P1 = movmean(P1,3);
%P1 = movmean(P1,3);
%P1 = movmean(P1,3);

% Remove some low frequency modes 
Y(1:round(length(Y)/50))=0;
Y(end-round(length(Y)/50)+2:end)=0;
Jaw1Dfiltered=ifft(Y);


figure(7)
plot(Jaw1Dfiltered)

% Fs=1;
% f = Fs*(0:(L/2))/L;
% ind=find(f>1/50);
% figure(6)
% plot(1./f(ind),P1(ind)) 
% title('Single-Sided Amplitude Spectrum of I(t)')
% xlabel('Spatial period (in pixels) ')
% ylabel('|P1(f)|')


%%
ToothSpace=9.5;
sigma=2;
x=-3*ToothSpace/2:3*ToothSpace/2;
ModelToothNoGap=exp(-x.^2/sigma^2)+exp(-(x-ToothSpace).^2/sigma^2)+exp(-(x+ToothSpace).^2/sigma^2);
ModelToothGap=exp(-(x-ToothSpace).^2/sigma^2)+exp(-(x+ToothSpace).^2/sigma^2);

ModelToothNoGap=ModelToothNoGap-mean(ModelToothNoGap);
ModelToothGap=ModelToothGap-mean(ModelToothGap);

figure(5)
hold on
plot(x+320,ModelToothNoGap+20)
plot(x+320,ModelToothGap+20)

%%
% WindowLength=length(x);
% for k=1:length(Jaw1Dfiltered)-WindowLength
%     Xwindow=Jaw1Dfiltered(k:k+WindowLength);
%     Xwindow.*ModelToothGap...
% end
%xcNoGap=xcorr(ModelToothNoGap,Jaw1Dfiltered);
%xcGap=xcorr(ModelToothGap,Jaw1Dfiltered);

figure(6)
clf
hold on
plot(Jaw1Dfiltered)

%plot(xcNoGap-xcGap)
%plot(xcGap)
%plot(xcNoGap-xcGap)