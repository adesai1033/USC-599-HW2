You read one page of a 3D computer vision paper and copy out the benchmark results that the paper
reports **for its own proposed method**. You get the page as an image and as extracted text. Use
the image to read the tables: the text loses the row and column layout. Use the text to read the
numbers exactly.

Paper: {title} (arXiv:{arxiv_id})

## What to extract

One result per (table row of the paper's own method) × (metric column) × (benchmark), for tables on
this page whose caption starts with "Table" or "Tab.".

- **benchmark**: the dataset, as named in the table or caption, e.g. "Mip-NeRF 360", "Tanks and
  Temples", "Deep Blending", "LLFF", "DTU", "Blender" (the NeRF synthetic scenes), "ScanNet",
  "Replica".
- **scene**: null when the number is an average over the dataset ("Avg.", "Mean", or a table that
  reports one number per dataset). Otherwise the scene name exactly as written ("bicycle", "scan24",
  "Truck").
- **metric**: one of PSNR, SSIM, LPIPS, Chamfer (Chamfer distance), F-score, Accuracy, Completeness,
  Training time, FPS, Memory. Skip every other metric.
- **value**: the number exactly as printed, as a number. Drop units and markers (bold, underline,
  †, *). Convert training time to minutes and memory to MB. Skip a cell that isn't a single number.
- **setting**: a short note on anything that changes comparability, taken from the table or caption.
  Examples: resolution ("1/4 resolution"), iteration count ("7K iterations"), the variant name when
  the table lists several versions of the method ("Ours-light"), or "reported in [ref]". Use null when
  nothing is stated.
- **table**: the table's label, e.g. "Tab. 1".

## Rules

- Only rows for the method this paper proposes. These are usually labeled "Ours", the method's
  name, or "full". Skip baselines and other papers' methods, even when the paper re-ran them. Skip
  ablation rows that remove a component ("w/o densification"). Keep variants the paper presents as
  its method (for example, both "Ours-7K" and "Ours-30K"), and tell them apart in `setting`.
- Copy numbers, never compute them. If an average isn't printed, don't make one from the scenes.
- Skip figures, plots, and numbers that appear only in running text.
- If the page has no such table, return an empty list.
