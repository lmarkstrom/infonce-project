# run_single_gmm.py 
=== Config: GMM input, d=256, N=256, epochs=100 ===
CV (norm coefficient of variation): 0.0279

Per-coordinate diagnostics (n_coords=256, n_samples=5000):
  AD avg statistic = 0.3764   AD pass rate = 96.1%  (<0.752)
  DP avg p-value   = 0.4938   DP pass rate = 97.3%  (>0.05)

Random-projection diagnostics (n_projections=100, n_samples=250):
  AD avg statistic = 0.4068   AD pass rate = 95.0%
  DP avg p-value   = 0.4859   DP pass rate = 92.0%

[Supervised backbone (512-dim, pre-classifier)]  d=512  CV (norm coefficient of variation) = 0.2402
  Per-coordinate: AD avg=21.3800 pass=0.0%   DP avg p=0.0000 pass=0.0%   [135 degenerate/zero-variance coords excluded from avg]
  Random-projection: AD avg=1.5328 pass=35.0%   DP avg p=0.1480 pass=41.0%
jovyan@jupyter-jelindeb:~/DD2610/infoNCE/infonce-project/src/replication/cif

# run_single_laplace.py 
=== Config: d=256, N=256, epochs=100 ===
CV (norm coefficient of variation): 0.0556

Per-coordinate diagnostics (n_coords=256, n_samples=5000):
  AD avg statistic = 0.3659   AD pass rate = 94.9%  (<0.752)
  DP avg p-value   = 0.5381   DP pass rate = 95.7%  (>0.05)

Random-projection diagnostics (n_projections=100, n_samples=250):
  AD avg statistic = 0.4172   AD pass rate = 93.0%
  DP avg p-value   = 0.5211   DP pass rate = 95.0%


# train_supervised.py
[Supervised backbone (512-dim, pre-classifier)]  d=512  CV (norm coefficient of variation) = 0.2402
  Per-coordinate: AD avg=21.3800 pass=0.0%   DP avg p=0.0000 pass=0.0%   [135 degenerate/zero-variance coords excluded from avg]
  Random-projection: AD avg=1.5328 pass=35.0%   DP avg p=0.1480 pass=41.0%
jovyan@jupyter-jelindeb:~/DD2610/infoNCE/infonce-project/src/replication/cif

# train_contrastive.py
[Contrastive backbone (512-dim, pre-projector)]  d=512  CV (norm coefficient of variation) = 0.1749
  Per-coordinate: AD avg=16.5974 pass=0.0%   DP avg p=0.0000 pass=0.0%   [2 degenerate/zero-variance coords excluded from avg]
  Random-projection: AD avg=0.6341 pass=75.0%   DP avg p=0.3317 pass=71.0%

[Projector output (128-dim, InfoNCE space)]  d=128  CV (norm coefficient of variation) = 0.1649
  Per-coordinate: AD avg=0.4700 pass=89.8%   DP avg p=0.4126 pass=85.9%
  Random-projection: AD avg=0.4428 pass=92.0%   DP avg p=0.4023 pass=84.0%

