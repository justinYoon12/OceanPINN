%% Load PINN generated pressure field
f_source=109; % Source frequency
filename='p_109_noiseless\\p_estimate.mat';
load(filename)
load("hotAndCold.mat") % For colormap

%% Plot magnitude
p_result_real=reshape(p_result(:,1),[length(z_test),length(r_test)]);
p_result_imag=reshape(p_result(:,2),[length(z_test),length(r_test)]);
k0=2*pi*f_source/1500;
p_estimate=(p_result_real+1i*p_result_imag).*besselh(0,2,k0*(transpose(r_test))); % Generated pressure field

figure
imagesc(r_test/1000,z_test,abs(p_estimate))
colorbar
colormap(jet)
title('Magnitude pressure')
xlabel('Range (km)')
ylabel('Depth (m)')
%% Plot Beamforming
d=transpose(z_test-z_test(1));
theta=linspace(-40*pi/180,40*pi/180,300);
b=zeros(length(theta),size(p_estimate,2));
for j=1:size(p_estimate,2)
    for i=1:length(theta)
        t_delay=d/1500*sin(theta(i));
        b(i,j)=exp(-1j*2*pi*f_source*t_delay)*p_estimate(:,j);
    end
    b(:,j)=b(:,j)/norm(b(:,j));
end

C_M=max(max(20*log10(abs(b))));

figure
imagesc(1:size(p_estimate,2),theta*180/pi,(20*log10(abs(b)))-C_M)
colorbar
clim([-20 0])
colormap(cmap)
ylabel('DOA (\circ)')
xlabel('Samples')

