# Cultivation circuits

The d=3 magic state cultivation circuits at p=0.001, with each T gate
replaced by `S`, so they are Clifford. They are test fixtures for
`StimEngine`.

- Work: "Data for 'Magic state cultivation: growing T states as cheap as
  CNOT gates'"
- Authors: Craig Gidney, Cody Jones, Noah Shutty
- DOI: <https://doi.org/10.5281/zenodo.13777072>
- Licence: CC-BY-4.0, <https://creativecommons.org/licenses/by/4.0/>

| File here | Original path inside `circuits.zip` |
|---|---|
| `inject_bell_d3_p0.001.stim` | `circuits/for_perfectionist_decoding/c=inject[bell]+cultivate,p=0.001,noise=uniform,g=css,q=14,b=Y,r=5,d1=3.stim` |
| `inject_teleport_d3_p0.001.stim` | `circuits/for_perfectionist_decoding/c=inject[teleport]+cultivate,p=0.001,noise=uniform,g=css,q=18,b=Y,r=6,d1=3.stim` |
| `inject_unitary_d3_p0.001.stim` | `circuits/for_perfectionist_decoding/c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim` |

Changes: the files are renamed; their contents are unchanged.
