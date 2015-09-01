function dpd_kolmogorov(dirloc, dirname)

begin_averaging = 50;

set(0,'defaultFigureVisible','off')

% some default values
Fkol = 1.0;
wnum = 1;
dt = 1e-3; % doesn't matter in viscosity computations
nt = 1000;

fileID = fopen('results.txt', 'w');

% dirname = 'numberdensity_10_dt_5e-4_kBT_0.0945_gammadpd_45_aij_1_F0_2.0_wavenum_1';
% filedir = sprintf('~/workspace/mounts/falcon/%s/h5', dirname);
filedir = sprintf('%s/%s/h5', dirloc, dirname);
pattern = sprintf('%s/flowfields-*.h5',filedir);

% parse dir name
c = strrep(dirname, '_', ' ');
c = textscan(c, '%s %f');
pname = c{1}; pval = c{2};

for i=1:length(pval)
    fprintf(fileID, '%s\t', pname{i});
end
fprintf(fileID, 'viscosity\tReynolds\n');
for i=1:length(pval)
    fprintf(fileID, '%g\t', pval(i));
end

for i=1:length(pval)
    if strcmp(pname{i}, 'F0')
        Fkol = pval(i);
    end
    if strcmp(pname{i}, 'wavenum')
        wnum = pval(i);
    end
    if strcmp(pname{i}, 'dt')
        dt = pval(i);
    end
end

% read data
listing = dir(pattern);

T = zeros(length(listing),1); % temperature

for i=1:length(listing)
    filename = listing(i).name;
    fullpath = sprintf('%s/%s',filedir,filename);
    
    u = squeeze(h5read(fullpath,'/u'));
    T(i) = sum(u(:).^2); % temperature
    
    if i == begin_averaging
        meanu = u;
    end
    if i > begin_averaging
        meanu = meanu + u;
    end
end

meanu = meanu ./ (length(listing)-begin_averaging+1); % average over time

meanu = mean(meanu,3); % average over z-axis
meanu = mean(meanu,1); % average over x-axis
meanu = squeeze(meanu);

% now meanu should be cos-like

fig = figure();
plot(dt*nt*(1:length(listing)), T, 'r-x');
title('Temperature')
xlabel('time')
ylabel('temperature')
saveas(fig, 'temperature.png')

% fit cos into the average velocity profile
L = length(meanu);
x = 1:L;

my_eval = @(x,p) p(1)*cos((2*pi*abs(p(2))/L) * x);
params = [max(meanu) wnum];
params = double(params);

xfit = double(x);
yfit = double(meanu);
opts = optimset('display','off');
params = lsqnonlin(@(t) my_eval(xfit,t)-yfit, params, [], [], opts);
yfit_new = my_eval(xfit,params);

% plot fit
fig = figure();
plot(xfit,yfit,'+r',xfit,yfit_new,'b');
text('units','pixels','position',[10 100],'fontsize',20,'string',...
    sprintf('%g ', params))
title('Fit for average velocity')
xlabel('x')
ylabel('average velocity')
saveas(fig, 'avg_velocity_fit.png')

vmax = max(my_eval(x,params));
nu = (Fkol*(L/wnum)^2)/(4*pi^2*vmax);
Re = vmax*(L/wnum)/nu;
fprintf(fileID, '%g\t\t%g\n',nu,Re);

% fprintf(fileID, '\nParameters of fitting function %s:\n',func2str(my_eval));
% fprintf(fileID, '%g\t',params);
% fprintf(fileID, '\n');

fclose(fileID);

end