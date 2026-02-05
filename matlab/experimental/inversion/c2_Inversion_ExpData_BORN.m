clear all
close all
clc

load DATA_scenario_exp_singletarget.mat
load DATA_object_exp_singletarget.mat
% load DATA_scenario_exp_twotargets.mat
% load DATA_object_exp_twotargets.mat


% ===============================
% DEFINITION OF USEFUL PARAMETERS
e0=8.85e-12;        %vacuum dielectric permittivity
m0=4*pi*1e-7;       %vacuum magnetic permeability
eb_eq=eb-1i*(sb/(e0*2*pi*freq));      %complex background permittivity
kb=2*pi*freq*sqrt(e0*m0*eb_eq);           %background wavenumber


%% =========================================================================================
% LINEAR INVERSION VIA TRUNCATED SINGULAR VALUE DECOMPOSITION (TSVD) AND BORN APPROXIMATION

Etot_approx_BORN=Einc_domain;
data_BORN=Escat;
S_BORN=kernel_scattering_exp(Etot_approx_BORN,Nx,Ny,lx,ly,1,eb,sb,freq,Nm,Rm);

[U,S,V]=svd(S_BORN,'econ');
S1=diag(S);
norm_sing_val=abs(S1)./(abs(S1(1)));

figure(1),clf,set(gcf,'color','w')
plot(20*log10(norm_sing_val),'b','linewidth',2),box on,grid on
xlabel('n'),title('Normalized Singular Values [dB]'),ylim([-80 1])
set(gca,'fontsize',12)

treashold_dB=input('Truncation treashold [dB]: ');
[~,Nt]=min(abs(20*log10(norm_sing_val)-treashold_dB));

figure(1),
hold on,plot(ones(1,100)*Nt,linspace(-100,treashold_dB,100),'--r','linewidth',1.5)
hold on,plot(linspace(0,Nt,100),ones(1,100)*treashold_dB,'--r','linewidth',1.5)
legend(['Truncation index: ' num2str(Nt) '@' num2str(treashold_dB) 'dB'])

PROF_rec_BORN=TSVD_solver(U,S,V,Nt,data_BORN,Nx,Ny);

figure(2),clf,set(gcf,'color','w')
subplot(1,3,1),imagesc(xvec/lambda0,yvec/lambda0,real(PROF_rec_BORN)),colorbar
xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title('Re[\tau]:reconstructed via BA')
axis xy, axis image,set(gca,'fontsize',14)
subplot(1,3,2),imagesc(xvec/lambda0,yvec/lambda0,imag(PROF_rec_BORN)),colorbar
xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title('Im[\tau]:reconstructed via BA')
axis xy, axis image,set(gca,'fontsize',14)
subplot(1,3,3),imagesc(xvec/lambda0,yvec/lambda0,abs(PROF_rec_BORN)/max(max(abs(PROF_rec_BORN)))),colorbar
hold on, contour(xvec/lambda0,yvec/lambda0,real(PROF),1,'--k','LineWidth',2)
xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title('Normalized Abs[\tau]:reconstructed via BA')
axis xy, axis image,set(gca,'fontsize',14)

if strcmp(dataset,'dielTM_dec8f.txt')

    figure(3),clf,set(gcf,'color','w'),hold on,box on,grid on
    subplot(1,2,1),plot(xvec,real(PROF(Ny/2,:)+1),'b','Linewidth',2)
    hold on,plot(xvec,real(PROF_rec_BORN(Ny/2,:)+1),'--r','Linewidth',2)
    hold on,plot(xvec,ones(1,Ny)*3.3,'-.r'), plot(xvec,ones(1,Ny)*2.7,'-.r')
    xlabel('x[m]'),title([{'Fresnel single diel target'};{['Re[\epsilon_x]: x-cut @ y=' num2str(y0)]};{'BA reconstruction'}]),axis square
    xlim([xvec(1) xvec(end)]),ylim([0.9 3.4])
    subplot(1,2,2),plot(yvec,real(PROF(:,43)+1),'b','Linewidth',2)
    hold on, plot(yvec,real(PROF_rec_BORN(:,43)+1),'--r','Linewidth',2)
    hold on,plot(xvec,ones(1,Ny)*3.3,'-.r'), plot(xvec,ones(1,Ny)*2.7,'-.r')
    xlabel('y[m]'),title([{'Fresnel single diel target'};{['Re[\epsilon_x]: y-cut @ x=' num2str(x0) 'm']};{'BA reconstruction'}]),axis square
    xlim([yvec(1) yvec(end)]),ylim([0.9 3.4])

elseif strcmp(dataset,'twodielTM_8f.txt')

    figure(3),clf,set(gcf,'color','w'),hold on,box on,grid on
    plot(xvec,real(PROF(39,:)+1),'b','Linewidth',2)
    hold on,plot(xvec,real(PROF_rec_BORN(39,:)+1),'--r','Linewidth',2)
    hold on,plot(xvec,ones(1,Ny)*3.3,'-.r'), plot(xvec,ones(1,Ny)*2.7,'-.r')
    xlabel('x[m]'),title([{'Fresnel two diel target'};{['Re[\epsilon_x]: x-cut @ y=' num2str(y0_l) 'm']};{'BA reconstruction'}]),axis square
    xlim([xvec(1) xvec(end)]),ylim([0.9 3.4])

end