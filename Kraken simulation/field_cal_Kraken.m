function [p_out, r,z] = field_cal_Kraken(f,NMEDIA,dep,ssp,NSD,ZS,NRD,rec_dep,NSR,ran)

fid=fopen(['fray_main_SWellEx96.env'],'wt');
fprintf(fid,"'SWellEx96'               ! TITLE\n" );
fprintf(fid,'%f                     ! FC (Hz) \n',f);
fprintf(fid,'%d                    ! NMEDIA \n', NMEDIA);
fprintf(fid,"'NVF'                ! SSPOPT (Analytic or C-linear interpolation)\n ");
fprintf(fid,'%d %d %d               ! DEPTH of bottom (m)\n',0,0,max(dep));

for ii=1:length(dep)
    fprintf(fid,'%d %f / \n',dep(ii), ssp(ii));
end
fprintf(fid,"0 0.0 240.00 \n");
fprintf(fid,"216.50 1572.30 0.0 1.760 0.2 0.0 / \n");
fprintf(fid,"240.00 1593.00 / \n");
fprintf(fid,"0 0.0 1040.00 \n");
fprintf(fid,"240.00 1881.00 0.0 2.060 0.060 0.0/ \n");
fprintf(fid,"1040.00 3245.00 / \n");
fprintf(fid,"'A' %d \n", 0.0);
fprintf(fid,"1040.00 5200.00 0.0 2.66 0.02/ \n");
% fprintf(fid,'%d %d %d %d %f / \n',max(dep), CP,CS,RHO_B, P_LOSS);
fprintf(fid,'%d %d         ! CLOW  CHIGH (m/s) \n', 0, 1650.00);
fprintf(fid, '%d           ! Rmax(km)\n', 0);
fprintf(fid,'%d                 ! NZS\n',NSD);
fprintf(fid,'%f /                ! SD(1:NSD) (m)\n',ZS);
fprintf(fid,'%d                 ! NRD\n',NRD);
for iii=1:NRD
    fprintf(fid,'%f ',rec_dep(iii));
end
fprintf(fid,'/               ! RD(1:NRD) (m)\n');
fclose(fid);


fid1=fopen(['field.flp'],'wt');
fprintf(fid1," 'fray_main_SWellEx96'               ! TITLE\n" );
fprintf(fid1,"'RA'               ! OPT 'X/R', 'C/A'\n" );
fprintf(fid1,"%d               ! M  (number of modes to include)'\n",9999 );
fprintf(fid1,"%d               ! NPROF\n",1 );
fprintf(fid1,"%f %f              ! RPROF(1:NPROF) (km)\n",0,100000 );
fprintf(fid1,"%d                 !NR\n", NSR);
for iii=1:NSR
    fprintf(fid1,'%f ',ran(iii)/1000);
end
fprintf(fid1,'/               ! R(1:NR)   (km)\n');
% fprintf(fid1,"%f %f /              ! R(1:NR)   (km)\n",RANGE1/1000, RANGE2/1000);
fprintf(fid1,"%d                !NSD\n",NSD);
fprintf(fid1,'%f /                ! SD(1:NSD) (m)\n',ZS);
fprintf(fid1,"%d                !NRD\n",NRD);
for iii=1:NRD
    fprintf(fid1,'%f ',rec_dep(iii));
end
fprintf(fid1,'/               ! RD(1:NRD) (m)\n');
fprintf(fid1,"%d                 !NRR\n",NRD);
fprintf(fid1,"0.000000 /               ! RR(1:NRR) (m)\n");
fclose(fid1);
tic
kraken 'fray_main_SWellEx96'
toc
[~, ~, ~, ~, POS, p]=read_shd('fray_main_SWellEx96.shd');
p_out=squeeze(p);
z=POS.r.depth;
r=POS.r.range;
% z=rec_dep;
% r=ran;