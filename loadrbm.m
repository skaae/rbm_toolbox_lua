function [ rbm ] = loadrbm( folder )
%LOADRBM load a RBM saved to CSV files
if nargin == 0
    folder = pwd();
end

fields = {'W','U','b','c','d',...
    'err_val','err_train','err_recon_train'};

rbm = struct();
for i = 1:numel(fields)
    field = fields{i};
    csv = ['rbm' field '.csv'];
    dat = csvread( fullfile(folder, csv) );
    rbm.(field) = dat;
end

end

