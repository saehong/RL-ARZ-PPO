%clear all
close all

r = r_vec_RL;
v = v_vec_RL;

 vm = 40; % m/s 100 miles per hour
 rm = 0.16; % 240 veh/mile /m
 rs = 0.115;
 Veq = @(rho) vm * (1 - rho/rm);
 vs = Veq(rs);

[M,N]= size(r);

r(:,N) = r(:,N-1);

%%
%plot of rs = 115 bkst outlet
% figure
% colormap([0  0  0]);
% subplot(1,2,1);
% surf(r,'FaceColor','white','EdgeColor','interp','MeshStyle','row');
% axis([0, N, 0, M, min(min(r))*0.95, max(max(r))*1.05]);
% ylabel('position x (km)','fontsize', 26)
% xlabel('time (min)','fontsize', 26)
% zlabel('Density (vehicle/km)','fontsize', 26)
% %title('rho','fontsize', 22)
% ax=gca;
%  ax.XTickMode = 'MANUAL';
%  ax.XTick = 0:(N-1)/4:N-1;
%  set(ax,'XTickLabel',[0,2,4,6,8],'fontsize', 22)
%  set(gca,'Ydir','reverse')
%  ax.YTickMode = 'MANUAL';
%  ax.YTick = 0:(M-1)/2:M-1;
%  set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
%  ax.ZTick = 0.10:0.01:0.130;
%  set(ax,'ZTickLabel',[100,110,120,130],'fontsize', 22)
%  hold on
%  x1=0:1:M-1;
%  plot3(0*ones(1,numel(x1)),x1,r(:,1),'-b','LineWidth',4);
%  hold on
%  x2=0:1:N-1;
%  plot3(x2,M*ones(1,numel(x2)),r(M, :),'-r','LineWidth',4);
%  hold on
% 
% 
% v= v*3.6;
% v(:,N) = v(:,N-1);
% colormap([0  0  0]); 
% subplot(1,2,2);
% surf(v,'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
% axis([0, N, 0, M, min(min(v))*0.95, max(max(v))*1.05]);
% ylabel('position x (km)','fontsize', 26)
% xlabel('time (min)','fontsize', 26)
% zlabel('Velocity (km/h)','fontsize', 26)
%  ax=gca;
%  ax.XTickMode = 'MANUAL';
%  ax.XTick = 0:(N-1)/4:N-1;
%  set(ax,'XTickLabel',[0,2,4,6,8],'fontsize', 22)
% set(gca,'Ydir','reverse')
%  ax.YTickMode = 'MANUAL';
%  ax.YTick = 0:(M-1)/2:M-1;
%  set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
%  ax.ZTick = 35:5:45;
%  set(ax,'ZTickLabel',[35,40,45],'fontsize', 22)
%  hold on
%  x1=0:1:M-1;
%  plot3(0*ones(1,length(x1)),x1,v(:,1),'-b','LineWidth',4);
%  hold on
%  x2=0:1: N-1;
%  plot3(x2,M*ones(1,numel(x2)),v(M,:),'-r','LineWidth',4);
%  hold on


%%
%plot of inlet rs = 125
figure
colormap([0  0  0]);
subplot(1,2,1);
surf(r,'FaceColor','white','EdgeColor','interp','MeshStyle','row');
axis([0, N, 0, M, min(min(r))*0.95, max(max(r))*1.05]);
ylabel('position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('Density (vehicle/km)','fontsize', 26)
ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/4:N-1;
 set(ax,'XTickLabel',[0,2,4,6,8],'fontsize', 22)
 %set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 0.10:0.01:0.140;
 set(ax,'ZTickLabel',[100,110,120,130,140],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,numel(x1)),x1,r(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:N-1;
 plot3(x2,0*ones(1,numel(x2)),r(1, :),'-r','LineWidth',4);

v= v*3.6;
v(:,N) = v(:,N-1);
colormap([0  0  0]); 
subplot(1,2,2);
surf(v,'FaceColor','white','edgecolor', 'interp','MeshStyle','row');
axis([0, N, 0, M, min(min(v))*0.95, max(max(v))*1.05]);
ylabel('position x (km)','fontsize', 26)
xlabel('time (min)','fontsize', 26)
zlabel('Velocity (km/h)','fontsize', 26)
 ax=gca;
 ax.XTickMode = 'MANUAL';
 ax.XTick = 0:(N-1)/4:N-1;
 set(ax,'XTickLabel',[0,2,4,6,8],'fontsize', 22)
%set(gca,'Ydir','reverse')
 ax.YTickMode = 'MANUAL';
 ax.YTick = 0:(M-1)/2:M-1;
 set(ax,'YTickLabel',[0,0.25,0.5],'fontsize', 22)
 ax.ZTick = 25:5:35;
 set(ax,'ZTickLabel',[25,30,35],'fontsize', 22)
 hold on
 x1=0:1:M-1;
 plot3(0*ones(1,length(x1)),x1,v(:,1),'-b','LineWidth',4);
 hold on
 x2=0:1:N-1;
 plot3(x2,0*ones(1,numel(x2)),v(1, :),'-r','LineWidth',4);



%%v = v/3.6;
 
%  figure
%  q_RL_control = r(2,1:N-1).* v(2,1:N-1);
%  plot(q_RL_control,'-b','LineWidth',4)
% % legend('inlet RL control','FontSize',22)
% hold on
% % plot(rs*vs*ones(1,N-1))
 
%% Reward= -sum((r - rs * ones(M,N)).^2/rs^2 ,1).^(1/2)- sum((v - vs * ones(M,N)).^2/vs^2,1).^(1/2) ;

% Reward = zeros(1,N);
% for k = 1:N
%     Reward(k) = -sqrt(sum((r(:,k) - rs * ones(M,1)).^2))/rs - sqrt(sum((v(:,k) - vs * ones(M,1)).^2))/vs;
% end
% 
% figure
% plot(Reward(1:N-1))
% ax=gca;
% ax.XTickMode = 'MANUAL';
% ax.XTick = 0:(N-1)/4:N-1;
% set(ax,'XTickLabel',[0,30,60,90,120])
% 
% sum(Reward(1:end-1),'all') 
% 
% save s_reward_RL_PI Reward



