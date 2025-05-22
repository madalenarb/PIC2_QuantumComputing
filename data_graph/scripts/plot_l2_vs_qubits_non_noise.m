% plot_l2_vs_qubits_non_noise.m
% L2‐error vs n_bits for all simulators & shot‐counts (no noise)

% load
T = readtable('data_csv/treated_data/qft_merged_no_noise.csv');

% unique targets & shots
targets = unique(T.target);
shots   = unique(T.shots);

figure; hold on;
markers = {'o','s','^','d','v','>','<'};
for i=1:numel(targets)
  for j=1:numel(shots)
    sel = strcmp(T.target,targets{i}) & T.shots==shots(j);
    if any(sel)
      x = T.n_bits(sel);
      y = T.l2_error(sel);
      plot(x, y, ...
           'DisplayName',sprintf('%s, shots=%d',targets{i},shots(j)), ...
           'Marker', markers{mod(i+j-2,numel(markers))+1}, ...
           'LineWidth',1.2);
    end
  end
end
xlabel('Number of Qubits (n\_bits)');
ylabel('L2 error');
title('Noiseless: L2 error vs n\_bits');
legend('Location','best'); grid on;
