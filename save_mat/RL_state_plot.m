clc;
clear all;
close all;

%% Backstepping - Outlet CASE1
BKST_Outlet = load('ARZ_Backstep_Outlet_Results.mat');

r = BKST_Outlet.r_vec_bcmk;
v = BKST_Outlet.v_vec_bcmk*3.6;

[M,N]= size(BKST_Outlet.r_vec_bcmk);


figure()
colormap([0  0  0]);
subplot(1,2,1);
surf(r(:, 1:N/2),'FaceColor','white','EdgeColor','interp','MeshStyle','row');
%axis([0, N/2, 0, M, 0.105, 0.135]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('Density (vehicle/km)','fontsize', 26)
%title('rho','fontsize', 22)
ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 0.105:0.01:0.135;
 set(ax,'ZTickLabel',[105,115,125,135],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,numel(x1)),x1,r(:,1),'-b','LineWidth',4);
%  hold on
%  x2=0:1:N/2-1;
%  plot3(x2, 0 *ones(1,numel(x2)),r(1, 1:N/2),'-r','LineWidth',4);
 hold on
 x2=0:1:N/2-1;
 plot3(x2, M *ones(1,numel(x2)),r(M, 1:N/2),'-r','LineWidth',4);
 
colormap([0  0  0]); 
subplot(1,2,2);
surf(v(:, 1:N/2),'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
%axis([0, N/2, 0, M, 32,40]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('velocity (km/h)','fontsize', 26)
%title('velocity','fontsize', 22)
 ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 32:2:40;
 set(ax,'ZTickLabel',[32,34,36,38,40],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,length(x1)),x1,v(:,1),'-b','LineWidth',4);
%  hold on
%  x2=0:1: N/2-1;
%  plot3(x2, 0*ones(1,numel(x2)),v(1, 1:N/2),'-r','LineWidth',4);
 hold on
 x2=0:1: N/2-1;
 plot3(x2, M*ones(1,numel(x2)),v(M, 1:N/2),'-r','LineWidth',4);

%% P-Control - Inlet CASE2


P_Inlet = load('ARZ_P_Control_Inlet_Results.mat');

r = P_Inlet.r_vec_base;
v = P_Inlet.v_vec_base*3.6;

[M,N]= size(P_Inlet.r_vec_base);


figure()
colormap([0  0  0]);
subplot(1,2,1);
surf(r(:, 1:N/2),'FaceColor','white','EdgeColor','interp','MeshStyle','row');
%axis([0, N/2, 0, M, 0.105, 0.135]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('Density (vehicle/km)','fontsize', 26)
%title('rho','fontsize', 22)
ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
%  set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 0.105:0.01:0.135;
 set(ax,'ZTickLabel',[105,115,125,135],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,numel(x1)),x1,r(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:N/2-1;
 plot3(x2, 0 *ones(1,numel(x2)),r(1, 1:N/2),'-r','LineWidth',4);
%  hold on
%  x2=0:1:N/2-1;
%  plot3(x2, M *ones(1,numel(x2)),r(M, 1:N/2),'-r','LineWidth',4);
 
colormap([0  0  0]); 
subplot(1,2,2);
surf(v(:, 1:N/2),'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
%axis([0, N/2, 0, M, 32,40]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('velocity (km/h)','fontsize', 26)
%title('velocity','fontsize', 22)
 ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
%  set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 32:2:40;
 set(ax,'ZTickLabel',[32,34,36,38,40],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,length(x1)),x1,v(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1: N/2-1;
 plot3(x2, 0*ones(1,numel(x2)),v(1, 1:N/2),'-r','LineWidth',4);
%  hold on
%  x2=0:1: N/2-1;
%  plot3(x2, M*ones(1,numel(x2)),v(M, 1:N/2),'-r','LineWidth',4);


%% PI Control - Outlet & Inlet CASE3


PI_Outlet_N_Inlet = load('ARZ_PI_Control_Outlet_N_Inlet_Results.mat');

r = PI_Outlet_N_Inlet.r_vec_base;
v = PI_Outlet_N_Inlet.v_vec_base*3.6;

[M,N]= size(PI_Outlet_N_Inlet.r_vec_base);


figure()
colormap([0  0  0]);
subplot(1,2,1);
surf(r(:, 1:N/2),'FaceColor','white','EdgeColor','interp','MeshStyle','row');
%axis([0, N/2, 0, M, 0.105, 0.135]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('Density (vehicle/km)','fontsize', 26)
%title('rho','fontsize', 22)
ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 0.105:0.01:0.135;
 set(ax,'ZTickLabel',[105,115,125,135],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,numel(x1)),x1,r(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:N/2-1;
 plot3(x2, 0 *ones(1,numel(x2)),r(1, 1:N/2),'-r','LineWidth',4);
 hold on
 x2=0:1:N/2-1;
 plot3(x2, M *ones(1,numel(x2)),r(M, 1:N/2),'-r','LineWidth',4);
 
colormap([0  0  0]); 
subplot(1,2,2);
surf(v(:, 1:N/2),'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
%axis([0, N/2, 0, M, 32,40]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('velocity (km/h)','fontsize', 26)
%title('velocity','fontsize', 22)
 ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 32:2:40;
 set(ax,'ZTickLabel',[32,34,36,38,40],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,length(x1)),x1,v(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1: N/2-1;
 plot3(x2, 0*ones(1,numel(x2)),v(1, 1:N/2),'-r','LineWidth',4);
 hold on
 x2=0:1: N/2-1;
 plot3(x2, M*ones(1,numel(x2)),v(M, 1:N/2),'-r','LineWidth',4);

%% RL - Outlet CASE1

RL_Outlet = load('ARZ_RL_Outlet_Results.mat');

r = RL_Outlet.r_vec_RL;
v = RL_Outlet.v_vec_RL*3.6;

[M,N]= size(RL_Outlet.r_vec_RL);


figure()
colormap([0  0  0]);
subplot(1,2,1);
surf(r(:, 1:N/2),'FaceColor','white','EdgeColor','interp','MeshStyle','row');
%axis([0, N/2, 0, M, 0.105, 0.135]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('Density (vehicle/km)','fontsize', 26)
%title('rho','fontsize', 22)
ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 0.105:0.01:0.135;
 set(ax,'ZTickLabel',[105,115,125,135],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,numel(x1)),x1,r(:,1),'-b','LineWidth',4);
%  hold on
%  x2=0:1:N/2-1;
%  plot3(x2, 0 *ones(1,numel(x2)),r(1, 1:N/2),'-r','LineWidth',4);
 hold on
 x2=0:1:N/2-1;
 plot3(x2, M *ones(1,numel(x2)),r(M, 1:N/2),'-r','LineWidth',4);
 
colormap([0  0  0]); 
subplot(1,2,2);
surf(v(:, 1:N/2),'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
%axis([0, N/2, 0, M, 32,40]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('velocity (km/h)','fontsize', 26)
%title('velocity','fontsize', 22)
 ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 32:2:40;
 set(ax,'ZTickLabel',[32,34,36,38,40],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,length(x1)),x1,v(:,1),'-b','LineWidth',4);
%  hold on
%  x2=0:1: N/2-1;
%  plot3(x2, 0*ones(1,numel(x2)),v(1, 1:N/2),'-r','LineWidth',4);
 hold on
 x2=0:1: N/2-1;
 plot3(x2, M*ones(1,numel(x2)),v(M, 1:N/2),'-r','LineWidth',4);


%% RL - Inlet CASE2

RL_Inlet = load('ARZ_RL_Inlet_Results.mat');

r = RL_Inlet.r_vec_RL;
v = RL_Inlet.v_vec_RL*3.6;

[M,N]= size(RL_Inlet.r_vec_RL);


figure()
colormap([0  0  0]);
subplot(1,2,1);
surf(r(:, 1:N/2),'FaceColor','white','EdgeColor','interp','MeshStyle','row');
%axis([0, N/2, 0, M, 0.105, 0.135]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('Density (vehicle/km)','fontsize', 26)
%title('rho','fontsize', 22)
ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
%  set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 0.105:0.01:0.135;
 set(ax,'ZTickLabel',[105,115,125,135],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,numel(x1)),x1,r(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:N/2-1;
 plot3(x2, 0 *ones(1,numel(x2)),r(1, 1:N/2),'-r','LineWidth',4);
%  hold on
%  x2=0:1:N/2-1;
%  plot3(x2, M *ones(1,numel(x2)),r(M, 1:N/2),'-r','LineWidth',4);
 
colormap([0  0  0]); 
subplot(1,2,2);
surf(v(:, 1:N/2),'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
%axis([0, N/2, 0, M, 32,40]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('velocity (km/h)','fontsize', 26)
%title('velocity','fontsize', 22)
 ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
%  set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 32:2:40;
 set(ax,'ZTickLabel',[32,34,36,38,40],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,length(x1)),x1,v(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1: N/2-1;
 plot3(x2, 0*ones(1,numel(x2)),v(1, 1:N/2),'-r','LineWidth',4);
%  hold on
%  x2=0:1: N/2-1;
%  plot3(x2, M*ones(1,numel(x2)),v(M, 1:N/2),'-r','LineWidth',4);

%% RL - Outlet & Inlet CASE3

RL_Outlet_N_Inlet = load('ARZ_RL_Outlet_N_Inlet_Results.mat');

r = RL_Outlet_N_Inlet.r_vec_RL;
v = RL_Outlet_N_Inlet.v_vec_RL*3.6;

[M,N]= size(RL_Outlet_N_Inlet.r_vec_RL);


figure()
colormap([0  0  0]);
subplot(1,2,1);
surf(r(:, 1:N/2),'FaceColor','white','EdgeColor','interp','MeshStyle','row');
%axis([0, N/2, 0, M, 0.105, 0.135]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('Density (vehicle/km)','fontsize', 26)
%title('rho','fontsize', 22)
ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 0.105:0.01:0.135;
 set(ax,'ZTickLabel',[105,115,125,135],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,numel(x1)),x1,r(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:N/2-1;
 plot3(x2, 0 *ones(1,numel(x2)),r(1, 1:N/2),'-r','LineWidth',4);
 hold on
 x2=0:1:N/2-1;
 plot3(x2, M *ones(1,numel(x2)),r(M, 1:N/2),'-r','LineWidth',4);
 
colormap([0  0  0]); 
subplot(1,2,2);
surf(v(:, 1:N/2),'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
%axis([0, N/2, 0, M, 32,40]);
ylabel('Position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('velocity (km/h)','fontsize', 26)
%title('velocity','fontsize', 22)
 ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/8:N-1;
 set(ax,'XTickLabel',[0,1,2,3,4],'fontsize', 22)
 set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 32:2:40;
 set(ax,'ZTickLabel',[32,34,36,38,40],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,length(x1)),x1,v(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1: N/2-1;
 plot3(x2, 0*ones(1,numel(x2)),v(1, 1:N/2),'-r','LineWidth',4);
 hold on
 x2=0:1: N/2-1;
 plot3(x2, M*ones(1,numel(x2)),v(M, 1:N/2),'-r','LineWidth',4);