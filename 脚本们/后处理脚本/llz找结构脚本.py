import functools
import os
import sys
import tarfile
from pathlib import Path

import ase.io
import numpy as np
from ase import Atom
from atomse.alter import del_atoms
from atomse.layers import (
    Layers,
    O3_Stacker,
    get_stacking_vectors,
    get_stacking_vectors_from_elmt,
)

try:
    from mathull import StructureEnumerator, StructureFilter, get_species
except ModuleNotFoundError as exc:
    mathull_home = os.environ.get("MATHULL_HOME")
    if mathull_home:
        mathull_home_path = str(Path(mathull_home).expanduser().resolve())
        if mathull_home_path not in sys.path:
            sys.path.insert(0, mathull_home_path)
        from mathull import StructureEnumerator, StructureFilter, get_species
    else:
        raise ModuleNotFoundError(
            "mathull not found. Install it in this venv or set MATHULL_HOME to its source root."
        ) from exc
def stack_layers(
    atoms, only_elmts, before_layers_cnt, stacking_vectors=None, mode="farthest"
):
    layers = Layers(atoms, only_elmts=only_elmts).layers
    if len(layers) == before_layers_cnt:
        stker = O3_Stacker(atoms, layers=layers)
        if stacking_vectors is None:
            _atoms = stker.stack_layer(0, range(len(layers))[1:], mode=mode)
        else:
            _atoms = stker.stack_layer_by_vectors(
                0, range(len(layers))[1:], vectors=stacking_vectors
            )
    else:
        idxs = []
        for idxs_in_layer in layers:
            idxs.extend(idxs_in_layer)
        _atoms = del_atoms(atoms, idxs)
    return _atoms


def get_operator(primitive_structure, elmt, layers, v_elmt):
    # 结构操作
    # structure_operator = None
    structure_operator = functools.partial(
        stack_layers,
        only_elmts=[
            # "Ni",
            # "Co",
            # "Mn",
            elmt,
            # 'H',
            # 'Ca',
        ],
        # ignore_elmts=['Li', 'O'],
        before_layers_cnt=len(layers),
        # stacking_vectors=get_stacking_vectors(primitive_structure, 16, [23, ]),
        # stacking_vectors=get_stacking_vectors_from_elmt(primitive_structure, v_elmt),
        mode="closest",
    )
    return structure_operator


def P2_stack(atoms):
    _atoms = atoms.copy()
    layers = Layers(_atoms, only_elmts=['Li', 'H']).layers
    if len(layers) < 2:
        _atoms = del_atoms(_atoms, 'Li')
    else:
        radian = np.radians(180)
        roate_matrix = np.array(
            [
                [np.cos(radian), -np.sin(radian), 0],
                [np.sin(radian), np.cos(radian), 0],
                [0, 0, 1],
            ]
        )
        for idx in layers[0]:
            atom = _atoms[idx]
            # nx = atom.position[1]
            # ny = atom.position[0]-atom.position[1]
            # nz = atom.position[2]+_atoms.cell.cartesian_positions([0,0,0.5])[2]
            # n_pos = [nx,ny,nz]
            n_pos = np.dot(roate_matrix, atom.position)
            n_pos += _atoms.cell.cartesian_positions([0, 0, 0.5])
            _atoms.append(Atom(atom.symbol, n_pos))
        del_idxs = []
        for idxs in layers[1:]:
            del_idxs.extend(idxs)
        _atoms = del_atoms(_atoms, del_idxs)
    return _atoms


# def get_operator(*args, **kwargs):
#     return P2_stack
def get_filter_combinations():
    f_c = (
        (
            "Ewald",
            {
                "num_outs": 5,
                "certain_charges": {
                    "Li": 1,
                    "Na": 1,
                    "K": 1,
                    "O": -2,
                    "X": 0,
                    "Bi": 3,
                    "P": 5,
                    "Cl": -1,
                    "S": -2,
                },
                "enhance_ewald_repulsion": {"Li": 2, "Na": 2, "K": 3, "Ca": 2},
                "charge_rule": (
                    {
                        "elmt": "Fe",
                        "ox_change": (2, 3),
                        "field": None,
                        "ignore_fields": ["T:4"],
                    },
                    {"elmt": "Fe", "ox_change": (2, 3), "field": "T:4"},
                    {
                        "elmt": "Mn",
                        "ox_change": (2, 3),
                        "field": None,
                        "ignore_fields": ["T:4"],
                    },
                    {"elmt": "Mn", "ox_change": (2, 3), "field": "T:4"},
                    {"elmt": "Ti", "ox_change": (3, 4), "field": None},
                    {"elmt": "Mn", "ox_change": (3, 4), "field": None},
                    {"elmt": "V", "ox_change": (3, 4), "field": None},
                    {"elmt": "V", "ox_change": (4, 5), "field": None},
                    {"elmt": "Cu", "ox_change": (2, 3), "field": None},
                    {"elmt": "Ni", "ox_change": (2, 3), "field": None},
                    {"elmt": "Ni", "ox_change": (3, 4), "field": None},
                    {"elmt": "Cr", "ox_change": (3, 4), "field": None},
                    {"elmt": "Co", "ox_change": (3, 4), "field": None},
                    {
                        "elmt": "Fe",
                        "ox_change": (3, 4),
                        "field": None,
                        "ignore_fields": ["T:4"],
                    },
                    {
                        "elmt": "Fe",
                        "ox_change": (4, 5),
                        "field": None,
                        "ignore_fields": ["T:4"],
                    },
                    {"elmt": "O", "ox_change": (-2, -1), "field": None},
                    {"elmt": "Fe", "ox_change": (3, 4), "field": "T:4"},
                ),
                "fields": {
                    "Li": "O:6",
                    "Ni": "O:6",
                    "Co": "O:6",
                    "Fe": "O:6",
                    "Mn": "O:6",
                    "Zn": "O:6",
                    "O": "O:6",
                    "Na": "O:6",
                },
                "ncores": 4,
            },
        ),
        # ("M3GNet", {"num_outs": 5, "relax": False}),
        # ("CHGNet", {"num_outs": 5, "relax": False}),
    )
    return f_c
def get_structures(primitive_structure, v_elmt):
    # 掺杂位置
    elmt = "Y"
    replace_elmts = ["Y", "Bi"]
    # replace_elmts = ["Ni", "Co", "Mn"]
    ref_elmt = ("Cl", 6)
    # species = get_species(primitive_structure, elmt, [elmt, "X"])
    species = get_species(primitive_structure, [elmt], [replace_elmts])
    # print(species)
    # 一层中的掺杂位置
    layers = Layers(
        primitive_structure,
        only_elmts=[
            elmt,
        ],
        spacing=1
    ).layers
    partial_species = get_species(primitive_structure, [layers[0]], [replace_elmts])
    # print(partial_species)
    # /////////////////////////////////////
    se_ = StructureEnumerator(
        primitive_structure=primitive_structure,
        species=species,
        ref_elmt=ref_elmt,
        symprec=0.1,
    )
    structures = se_.enumerate(
        # sizes=range(1),
        sizes=(1,),
        partial_species=partial_species,
        # partial_species=species,
        # structure_operator=None,
        # structure_operator=get_operator(),
        # structure_operator=get_operator(primitive_structure, elmt, layers, v_elmt),
        limit_latticeC=True,
        # limit_concentration={
        # "Ca": (1/9, 1.5/9),
        # "Ni": (1/9, 1.5/9),
        # "Co": (1/9, 1.5/9),
        # "Mn": (1/9, 1.5/9),
        # "Zn": (2 / 108, 2.5 / 108),
        # },
        ncores=8,
        verbose=True,
    )
    # print(structures)
    structures_dict = se_.group_by_concentration(
        structures,
        # elmt_symbols=('Li',),
        elmt_symbols=('Y', 'Bi'),
        # elmt_symbols=("Li", "Ni", "Co", "Mn", "O"),
        # cutoffs={2: 7, 3: 5},
        deduplication=True,
        prefer_size="smaller",
    )
    # return structures_dict
    sf_ = StructureFilter(filter_combinations=get_filter_combinations())
    filtered_structures_dict = sf_.filter_group_structures(structures_dict)
    return filtered_structures_dict
def write_structures(poscar, outdir, v_elmt):
    poscar = Path(poscar)
    primitive_structure = ase.io.read(poscar)
    structures_dict = get_structures(primitive_structure, v_elmt)
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    for symbol_, structures in structures_dict.items():
        for cnt, structure in enumerate(structures, start=1):
            ase.io.write(
                outdir / f"{symbol_}_{cnt:03d}.vasp", structure, direct=True, sort=True
            )
    print("Done!")


def resolve_base_dir():
    env_path = os.environ.get("ZJY_HOME") or os.environ.get("LLZ_HOME")
    if not env_path:
        return Path(".").resolve()

    p = Path(env_path).expanduser().resolve()
    if p.is_dir():
        return p

    # Allow env var to point to archive like zjy.tar.gz.
    p_str = str(p).lower()
    if p.is_file() and p_str.endswith(".tar.gz"):
        extract_parent = p.parent
        target_dir = extract_parent / p.name[:-7]
        if not target_dir.exists():
            with tarfile.open(p, mode="r:gz") as tf:
                tf.extractall(path=extract_parent)
        return target_dir

    return Path(".").resolve()


base_dir = resolve_base_dir()
candidate_poscars = [
    base_dir / "LYC2-CONTCAR",
    base_dir / "Li3YCl6.vasp",
    Path("./Li3YCl6.vasp"),
]
poscar = next((p for p in candidate_poscars if p.exists()), None)
if poscar is None:
    raise FileNotFoundError(
        "No POSCAR found. Set ZJY_HOME/LLZ_HOME or place Li3YCl6.vasp in current directory."
    )

# poscar = Path(r'd:\SuL04\Downloads\NaMnO2-3.vasp')
outdir = base_dir / "Bidoping"
# outdir = poscar.parent / "same_layer_max_to_13"
v_elmt = "Y"
write_structures(poscar, outdir, v_elmt)







