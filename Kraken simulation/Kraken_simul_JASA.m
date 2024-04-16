%% parameters for Kraken simulation
NMEDIA=3;
f_source=109; %Frequency
f=f_source;

ran=1500:20:3500;
[~,ran_sample_ind]=find(ran<=2100|(ran>=2300&ran<=2700)|ran>=2900); % Sampled range
ran=ran(ran_sample_ind);
ZS=9; % Source depth (SWellEx-96)

VLA_dep=linspace(94.125,212.25,64); % SwellEx-96 VLA sensor depth
dep_sample_ind=linspace(1,61,13);
rec_dep=VLA_dep(dep_sample_ind);

NSR=length(ran);
NSD=1;
NRD=length(rec_dep);

% Options for noise 
noise_opt=1; %0: Add noise, other:noiseless
snr=12;
%% Ssp load
load('ssp_SWellex96.mat');

dep_save=linspace(0,dep(end),201);
SSP_save=spline(dep,ssp,dep_save);

%% Kraken simulation
[p_replica,r,z]=field_cal_Kraken(f_source,NMEDIA,dep_save,SSP_save,NSD,ZS,NRD,rec_dep,NSR,ran);

p_real=real(p_replica);
p_imag=imag(p_replica);
p_abs=abs(p_replica);

%% Add noise
if noise_opt==0
    src_p=norm(p_replica,"fro")/sqrt(numel(p_replica));
    noise_p=10^(-snr/20)*src_p;
    p_noise=noise_p*(randn(size(p_replica,1),size(p_replica,2))+1i*randn(size(p_replica,1),size(p_replica,2)))/sqrt(2);
    snr_cal=20*log10(norm(p_replica,"fro")/norm(p_noise,"fro"))
    p_replica=p_replica+p_noise;
    p_real=real(p_replica);
    p_imag=imag(p_replica);
end

%% Transform
k0=2*pi*f_source/1500;
p_pe=p_replica./besselh(0,2,k0*transpose(r));
p_pe_real=real(p_pe);
p_pe_imag=imag(p_pe);
div_temp=transpose(sqrt(2./(pi*k0*r)));
p_pe_abs=abs(p_replica./div_temp);

%%  Save the file
if noise_opt==0
    file_name=['p_' num2str(f_source) '_ran' num2str(length(r)) '_' num2str(min(r)) '_' num2str(max(r)) '_dep' num2str(length(z)) '_' num2str(floor(min(z))) '_' num2str(floor(max(z))) '_snr' num2str(snr) '_mag'];
else
    file_name=['p_' num2str(f_source) '_ran' num2str(length(r)) '_' num2str(min(r)) '_' num2str(max(r)) '_dep' num2str(length(z)) '_' num2str(floor(min(z))) '_' num2str(floor(max(z))) '_mag'];
end
r=transpose(r);
z=transpose(z);
% save(file_name,"r","z","p_abs","p_imag","p_real","f","SSP_save","dep_save","p_pe_abs","p_pe_real","p_pe_imag")