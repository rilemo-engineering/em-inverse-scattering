clear all
close all
clc

%load DATA_scenario.mat
load DATA_scenario_square.mat
%load DATA_scenario_noweak.mat
AL=0; %if AL=1 -> aspect limited measurement configuration. what is the effect if i see my region of interest only from 1 side?. in most of exercises in inverse scattering prob i put antennas around target, but in some cases we can place antennas only in one side. we can see this occurrence by changing the param here
% ===============================
% DEFINITION OF USEFUL PARAMETERS
e0=8.85e-12;        %vacuum dielectric permittivity
m0=4*pi*1e-7;       %vacuum magnetic permeability
eb_eq=eb-1i*(sb/(e0*2*pi*freq));      %complex background permittivity
kb=2*pi*freq*sqrt(e0*m0*eb_eq);           %background wavenumber


% ================================================
% ADD WHITE GAUSSIAN NOISE ON SCATTERED FIELD DATA
SNR=30;
Escat=awgn(Escat,SNR,'measured',345);

%% =========================================================================================
% LINEAR INVERSION VIA TRUNCATED SINGULAR VALUE DECOMPOSITION (TSVD) AND BORN APPROXIMATION

Etot_approx_BORN=Einc_domain;
data_BORN=Escat;
S_BORN=kernel_scattering(Etot_approx_BORN,Nx,Ny,lx,ly,1,eb,sb,freq,Nm,Rm);

if AL==0
[U,S,V]=svd(S_BORN); 
S1=diag(S); 
norm_sing_val=abs(S1)./(abs(S1(1)));

figure(1),clf,set(gcf,'color','w')
plot(20*log10(norm_sing_val),'b','linewidth',2),box on,grid on
xlabel('n'),title('Normalized Singular Values [dB]'),ylim([-80 1])
set(gca,'fontsize',12)

treashold_dB=input('Truncation treashold [dB]: '); % truncation effect
[~,Nt]=min(abs(20*log10(norm_sing_val)-treashold_dB));

fprintf('\nTruncation index: %d\n', Nt);

figure(1),
hold on,plot(ones(1,100)*Nt,linspace(-100,treashold_dB,100),'--r','linewidth',1.5)
hold on,plot(linspace(0,Nt,100),ones(1,100)*treashold_dB,'--r','linewidth',1.5)
legend(['Truncation index: ' num2str(Nt) '@' num2str(treashold_dB) 'dB'])

PROF_rec_BORN=TSVD_solver(U,S,V,Nt,data_BORN,Nx,Ny);

mm_r=min([min(min(real(PROF))) min(min(real(PROF_rec_BORN)))]);
mm_i=min([min(min(imag(PROF))) min(min(imag(PROF_rec_BORN)))]);
MM_r=max([max(max(real(PROF))) max(max(real(PROF_rec_BORN)))]);
MM_i=max([max(max(imag(PROF))) max(max(imag(PROF_rec_BORN)))]);

figure(2),clf,set(gcf,'color','w')
subplot(2,2,1),imagesc(xvec/lambda0,yvec/lambda0,real(PROF)),colorbar
xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title('Re[\tau]:actual')
axis xy, axis image,caxis([mm_r MM_r])
subplot(2,2,2),imagesc(xvec/lambda0,yvec/lambda0,imag(PROF)),colorbar
xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title('Im[\tau]:actual')
axis xy, axis image,caxis([mm_i MM_i])
subplot(2,2,3),imagesc(xvec/lambda0,yvec/lambda0,real(PROF_rec_BORN)),colorbar
xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title('Re[\tau]:reconstructed via BA')
axis xy, axis image,caxis([mm_r MM_r])
subplot(2,2,4),imagesc(xvec/lambda0,yvec/lambda0,imag(PROF_rec_BORN)),colorbar
xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title('Im[\tau]:reconstructed via BA')
axis xy, axis image,caxis([mm_i MM_i])

NMSE_BORN=sum(sum(abs(PROF-PROF_rec_BORN).^2))/sum(sum(abs(PROF).^2))


figure(3),clf,set(gcf,'color','w')
imagesc(xvec/lambda0,yvec/lambda0,abs(PROF_rec_BORN)/max(max(abs(PROF_rec_BORN)))),colorbar
hold on, contour(xvec/lambda0,yvec/lambda0,real(PROF),1,'--k','LineWidth',2)
xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title('Normalized Abs[\tau]:reconstructed via BA')
axis xy, axis image
set(gca,'fontsize',22)



%% ========================== %%
% ASPECT LIMITATION 
 
 else

mask1=zeros(Nm,Nv);
mask1(2:10,2:10)=1;
data_BORN_AL_1=data_BORN.*mask1; %upper arc
S_BORN_AL_1=S_BORN.*reshape(repmat(mask1,[1 1 Nx*Ny]),Nm*Nv,Nx*Ny);


figure(4),clf,set(gcf,'color','w')
subplot(1,2,1),imagesc(1:1:Nm,1:1:Nv,abs(data_BORN)),colorbar,axis square,axis xy
xlabel('Tx'), ylabel('Rx'),title([{'Full-aspect'} {'data matrix'}])
subplot(1,2,2),imagesc(1:1:Nm,1:1:Nv,abs(data_BORN_AL_1)),colorbar,axis square,axis xy
xlabel('Tx'), ylabel('Rx'),title([{'Aspect-limited'} {'data matrix, upper arc'}])

% ========================== %%
% RECONSTRUCTION IN ASPECT LIMITATION 

fprintf('ASPECT LIMITATION 1: UPPER ARC\n')
[U1,S,V1]=svd(S_BORN_AL_1);
S1=diag(S);
norm_sing_val=abs(S1)./(abs(S1(1)));

figure(5),clf,set(gcf,'color','w')
plot(20*log10(norm_sing_val),'b','linewidth',2),box on,grid on
xlabel('n'),title('Normalized Singular Values [dB]'),ylim([-80 1])
set(gca,'fontsize',12)

treashold_dB=input('Truncation treashold [dB]: ');
[~,Nt1]=min(abs(20*log10(norm_sing_val)-treashold_dB));
fprintf('\nTruncation index: %d\n', Nt1);

figure(5),
hold on,plot(ones(1,100)*Nt1,linspace(-100,treashold_dB,100),'--r','linewidth',1.5)
hold on,plot(linspace(0,Nt1,100),ones(1,100)*treashold_dB,'--r','linewidth',1.5)
legend(['Truncation index: ' num2str(Nt1) '@' num2str(treashold_dB) 'dB'])
pause(1)

PROF_rec_BORN_AL_1=TSVD_solver(U1,S,V1,Nt1,data_BORN_AL_1,Nx,Ny);


figure(6),clf,set(gcf,'color','w')
imagesc(xvec/lambda0,yvec/lambda0,abs(PROF_rec_BORN_AL_1)/max(max(abs(PROF_rec_BORN_AL_1)))),colorbar
hold on, contour(xvec/lambda0,yvec/lambda0,real(PROF),1,'--k','LineWidth',2)
xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title([{'Normalized Abs[\tau]:'} {'reconstructed via BA and Asp.Lim.1'}])
axis xy, axis image
set(gca,'fontsize',22)
pause(1)

NMSE_BORN_AL_1=sum(sum(abs(PROF-PROF_rec_BORN_AL_1).^2))/sum(sum(abs(PROF).^2))

 end