close all
clear all
clc

% BEFORE RUNNING THE CODE, TAKE NOTE ABOUT USEFUL DATA INFORMATION
% - freq: working frequency [MHz]
% - eb: permittivity of the background
% - sb: conductivity of the background
% - lx: x-dimension of the Domain of Investigation [m]
% - ly: y-dimension of the Domain of Investigation [m]
% - Rm: radius of the measurement circular surface [m]
% - Nm: number of measurement point, i.e., DoF=2*beta*a
%       In this case, a=sqrt(2)*lx/2.
%       NB. Round up the DoF number
% - Nv: number of illumination directions. Usually, Nv=Nm.
% - ex: higher permittivity in your profile
% - sx: higher conductivity in your profile
% - targets shape, dimension and position

n_iter=1000;
[Escat, PROF, Einc_domain, Etot_domain, freq, lx, ly, eb, sb, Nx, Ny, Rm, DOF]=forward_solver(n_iter);


% OBSERVING FORWARD SOLVER OUTPUTS
% - nx: number of discretization cells for x-dimension
% - ny: number of discretization cells for y-dimension
% - Escat: multiview-multistatic data matrix. Dimension: Nm x Nv
% - PROF: contrast profile. Dimension: Ny x Nx
% - Einc_domain: incident field on the RoI. Dimension: Ny x Nx x Nv
% - Etot_domain: actual total field on the RoI. Dimension: Ny x Nx x Nv

lambda0=3*1e8/freq;
Nm=size(Escat,1);
Nv=size(Einc_domain,3);
dx=lx/Nx;
dy=ly/Ny;
xvec=-lx/2+dx/2:dx:lx/2-dx/2;
yvec=-ly/2+dy/2:dy:ly/2-dy/2;
[X,Y]=meshgrid(xvec,yvec);
meas_pos_theta=linspace(0,2*pi-2*pi/Nm,Nm);

visualization=input('Would you like to visualize fields and profile? Yes[1] No[0]: ');
if visualization==1
    
    figure(1),clf,set(gcf,'color','w'),hold on,box on,grid on
    imagesc(xvec,yvec,abs(PROF)),colormap(flipud(gray))
    plot(xvec,ones(1,Ny)*lx/2,'k')
    plot(Rm*cos(linspace(0,2*pi,100)),Rm*sin(linspace(0,2*pi,100)),'--k')
    plot(Rm*cos(meas_pos_theta),Rm*sin(meas_pos_theta),'.r','markersize',20)
    plot(xvec,-ones(1,Ny)*lx/2,'k')
    plot(ones(1,Nx)*ly/2,yvec,'k')
    plot(-ones(1,Nx)*ly/2,yvec,'k')
    xlabel('x[m]'),ylabel('y[m]'),title('Simulated Scenario')
    axis xy, axis image
    legend('RoI','Measurement Surface','Measurement Points')
    legend('location','best')
    
    pause(1)
    
    figure(2),clf,set(gcf,'color','w')
    subplot(1,2,1),imagesc(xvec/lambda0,yvec/lambda0,real(PROF)),colorbar
    xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title('Re[\tau]')
    axis xy, axis image
    subplot(1,2,2),imagesc(xvec/lambda0,yvec/lambda0,imag(PROF)),colorbar
    xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title('Im[\tau]')
    axis xy, axis image
    
    pause(1)
    
    figure(3),clf,set(gcf,'color','w')
    imagesc(1:1:Nv,1:1:Nm,abs(Escat)),colorbar
    xlabel('nv'),ylabel('nm'),title([{'MVMS Data Matrix'};{'(amplitude)'}])
    axis xy, axis image
    
    pause(1)
    
    figure(4),clf,set(gcf,'color','w')
    for kv=1:Nv
        subplot(1,2,1),imagesc(xvec/lambda0,yvec/lambda0,abs(Einc_domain(:,:,kv))),colorbar
        xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title([{'Amplitude of Incident Field'};{['[nv=' num2str(kv) ']']}])
        axis xy, axis image
        subplot(1,2,2),imagesc(xvec/lambda0,yvec/lambda0,angle(Einc_domain(:,:,kv))),colorbar
        xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title([{'Phase of Incident Field'};{['[nv=' num2str(kv) ']']}])
        axis xy, axis image
        pause(0.5)
    end
%     
%     pause(1)
%     
%     figure(5),clf,set(gcf,'color','w')
%     for kv=1:Nv
%         subplot(1,2,1),imagesc(xvec/lambda0,yvec/lambda0,abs(Etot_domain(:,:,kv))),colorbar
%         xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title([{'Amplitude of Total Field'};{['[nv=' num2str(kv) ']']}])
%         axis xy, axis image
%         subplot(1,2,2),imagesc(xvec/lambda0,yvec/lambda0,angle(Etot_domain(:,:,kv))),colorbar
%         xlabel('x/\lambda_0'),ylabel('y/\lambda_0'),title([{'Phase of Total Field'};{['[nv=' num2str(kv) ']']}])
%         axis xy, axis image
%         pause(0.5)
%     end
    
    
end



clear n_iter

save DATA_scenario.mat

