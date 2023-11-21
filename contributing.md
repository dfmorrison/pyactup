Generally useful adjustments and additions to PyACTUp are welcome. Please clone
the repository, make a suitable branch, and when finished make merge request on GitHub in the usual way,
to ensure review and, if necessary, discussion of your changes.
Before making a merge request, please ensure

- that *all* the unit tests pass

- that suitable unit tests have been added for any additions you make, or to test for regressions of any bugs
  you have fixed

- and that the documentation has been suitably revised, if necessary, and *builds correctly*.

In addition, PyACTUp is at the heart of PyIBL. Please *always* also clone the PyIBL repository and ensure that PyIBL’s
unit tests all pass with your modifications to PyACTUp, no matter how trivial they may seem.
The `-e` option to `pip install` can be useful for ensuring your new version of PyACTUp is used by PyIBL.

Depending upon the changes you are making, some further things to consider:

- Should your changes also be reflected in the `activationₕistory`?

- Do your changes work correctly both with and without indexing?

- Do any API changes closely match the current API choices?

- Do all undocumented or otherwise non-public functions, methods or other members’ names start with an underscore?

- When adding a new function, method or similar, not only do you need to include a suitable docstring in the definition
  itself, it needs to be cited in `doc/index.rst`

- If you need to import any modules not already imported by PyACTUp, please be sure to update `setup.py`
  and `requirements.txt` to match.

- Ensure appropriate errors, worded in ways helpful to the end user, are raised, and document all errors
  that can be raised in the docstring.

- Do your changes need to be carefully tested with `optimized_learning`, `forget()`, `fixed_noise`
  and/or `use_actr_similarity`?
