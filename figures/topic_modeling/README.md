# NMF K=8 Topic Labels

This document lists the topic labels for both preprocessing variants (baseline and natural language) at **K = 8**. These labels were assigned by manually inspecting the top-15 terms and representative questions for each topic.

---

## 1. Baseline (keep code)

| Topic | Label | Top Terms |
|---|---|---|
| Topic 0 | Python basics | list, print, string, function, number, value, input, like, want, range, return, code, len, dictionary |
| Topic 1 | Error / environment | py line, lib, py, line, packages, file, python2, site packages, lib python2, site, usr, traceback, error, recent, traceback recent |
| Topic 2 | Django | models, model, django, charfield, models model, class, max_length, models charfield, charfield max_length, foreignkey, true, models foreignkey, objects, field, user |
| Topic 3 | Object-oriented programming | self, def, class, __init__, __init__self, def__init__, self self, return, object, method, print, import, parent, object def, button |
| Topic 4 | Pandas | dataframe, pandas, df, column, columns, pd, data, row, 00, csv, index, 12, 10, 01, rows |
| Topic 5 | NumPy / Matplotlib | np, array, numpy, plt, plot, matplotlib, import, import numpy, numpy np, image, np array, numpy array, axis, data, shape |
| Topic 6 | File / script operations | python, file, script, os, files, run, using, path, import, command, use, open, module, directory, running |
| Topic 7 | Web requests | url, request, html, page, post, data, response, requests, app, django, server, form, content, json, api |

---

## 2. Natural Language (remove code)


| Topic | Label | Top Terms |
|---|---|---|
| Topic 0 | General coding questions | code, python, using, use, know, work, like, just, need, ve, program, way, make, problem, time |
| Topic 1 | Data structures | list, dictionary, lists, string, values, value, element, want, elements, like, strings, python, key, output, loop |
| Topic 2 | Django | django, py, model, form, template, user, app, models, view, field, page, views, html, database, admin |
| Topic 3 | File / CSV / XML handling | file, text, files, text file, csv, line, read, csv file, txt, write, xml, lines, output, data, file python |
| Topic 4 | Pandas | dataframe, pandas, column, columns, data, values, row, rows, pandas dataframe, array, like, numpy, index, value, want |
| Topic 5 | Error messages | error, following, getting, code, following error, trying, getting error, message, error message, wrong, help, try, following code, gives, using |
| Topic 6 | Environment / installation | script, run, python, install, command, installed, module, python script, import, py, windows, running, package, directory, version |
| Topic 7 | Object-oriented programming | function, class, method, object, instance, return, functions, variable, called, value, way, python, like, classes, attribute |



### Key Differences

- **Baseline** produces topics organized primarily by **technology stack** (NumPy, Matplotlib, web requests), because code-related terms such as `np`, `plt`, `url`, and `requests` are retained.
- **Natural language** produces topics organized primarily by **problem type** (data structures, error messages, environment), because code blocks are removed and only natural-language content remains.
- Topics that appear in both variants: Django (Topic 2), Pandas (Topic 4), Object-oriented programming.
- Topics unique to baseline: NumPy / Matplotlib, Web requests.
- Topics unique to natural language: General coding questions, Data structures, File / CSV / XML handling, Error messages, Environment / installation.

---

