# Visualization

Intent plotting code, aggregate data and figures were separated by function; numerical inputs and plotting logic are unchanged.

From the repository root:

```sh
python scripts/visualization/make_intent_figures.py
```

Requires NumPy and matplotlib; no GPU, model download or inference is triggered. Reads [aggregate counts](../../tables/intent_annotation/figure_data.json) and writes to [intent figures](../../figures/intent_annotation/). Use `--data PATH --out PATH` to plot another aggregate file without overwriting the published outputs.

The original raw-log hashes in the JSON are retained as historical provenance; they are not current repository paths. The two figures measure output validation, synthetic diagnostic behavior and prompt sensitivity, not real-question accuracy.
