close all
clear all
clc

eb=1;
sb=0;

freq=4*1e9;    %% data frequency
lambda0=3*1e8/freq;
lx=0.15;       %% side [m] of the imaging (or investigated) domain
ly=lx;

Nx=64;         %% number of pixel along x-direction
Ny=64;         %% number of pixel along y-direction

dataset='dielTM_dec8f.txt';
%dataset='twodielTM_8f.txt';
[Escat, Einc_domain]=load_data_fr2001(freq,dataset,Nx,Ny);   %%% load the scattered and incident field data provided by the dataset

Nm=size(Escat,1);     %% number of receivers
Nv=size(Escat,2);     %% number of transmitters
Rv=0.72135;           %% distance from the tx and the center of the investigated domain
Rm=0.76135;           %% distance from the rx and the center of the investigated domain
dx=lx/Nx;
dy=ly/Ny;
xvec=-lx/2+dx/2:dx:lx/2-dx/2;
yvec=-ly/2+dy/2:dy:ly/2-dy/2;
[X,Y]=meshgrid(xvec,yvec);

if strcmp(dataset,'dielTM_dec8f.txt')
    save DATA_scenario_exp_singletarget.mat


    %% FRESNEL DIELECTRIC SINGLE TARGET PROFILE (to appraise the inversion results)

    r0=0.015;            % circle radius
    x0=0.025;           % center of circle: x-coordinate
    y0=0.0;             % center of circle: y-coordinate
    rxy=sqrt((X-x0).^2+(Y-y0).^2);
    PROF=zeros(Ny,Nx);
    PROF(rxy<=r0)=2;


    figure(1),clf,set(gcf,'color','w'),hold on,box on,grid on
    subplot(1,2,1),imagesc(xvec,yvec,real(PROF)),colorbar
    xlabel('x[m]'),ylabel('y[m]'),title([{'Fresnel single diel target'};{'Re[\tau]'}])
    axis xy, axis image
    subplot(1,2,2),imagesc(xvec,yvec,imag(PROF)),colorbar
    xlabel('x[m]'),ylabel('y[m]'),title([{'Fresnel single diel target'};{'Im[\tau]'}])
    axis xy, axis image


    figure(2),clf,set(gcf,'color','w'),hold on,box on,grid on
    subplot(1,2,1),plot(xvec,real(PROF(Ny/2,:)),'b','Linewidth',2)
    xlabel('x[m]'),title([{'Fresnel single diel target'};{['Re[\tau]: x-cut @ y=' num2str(y0)]}]),axis square
    subplot(1,2,2),plot(xvec,real(PROF(:,43)),'b','Linewidth',2)
    xlabel('y[m]'),title([{'Fresnel single diel target'};{['Re[\tau]: y-cut @ x=' num2str(x0) 'm']}]),axis square


    save('DATA_object_exp_singletarget.mat','r0','x0','y0','PROF')

elseif strcmp(dataset,'twodielTM_8f.txt')

    save DATA_scenario_exp_twotargets.mat

    %% FRESNEL DIELECTRIC TWO TARGET PROFILE (to appraise the inversion results)

    % leftmost cylinder
    r0=0.015;            % circle radius
    x0_l=-0.045;          % center of circle: x-coordinate
    y0_l=0.015;           % center of circle: y-coordinate
    rxy=sqrt((X-x0_l).^2+(Y-y0_l).^2);
    PROF=zeros(Ny,Nx);
    PROF(rxy<=r0)=2;

    % rightmost cylinder
    x0_r=0.045;
    y0_r=0.005;
    rxy=sqrt((X-x0_r).^2+(Y-y0_r).^2);
    PROF(rxy<=r0)=2;



    figure(1),clf,set(gcf,'color','w'),hold on,box on,grid on
    subplot(1,2,1),imagesc(xvec,yvec,real(PROF)),colorbar
    xlabel('x[m]'),ylabel('y[m]'),title([{'Fresnel two diel target'};{'Re[\tau]'}])
    axis xy, axis image
    subplot(1,2,2),imagesc(xvec,yvec,imag(PROF)),colorbar
    xlabel('x[m]'),ylabel('y[m]'),title([{'Fresnel two diel target'};{'Im[\tau]'}])
    axis xy, axis image


    figure(2),clf,set(gcf,'color','w'),hold on,box on,grid on
    plot(xvec,real(PROF(39,:)),'b','Linewidth',2)
    xlabel('x[m]'),title([{'Fresnel two diel target'};{['Re[\tau]: x-cut @ y=' num2str(y0_l)]}]),axis square


    save('DATA_object_exp_twotargets.mat','r0','x0_l','x0_r','y0_l','y0_r','PROF')
end