#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: lilz
# @Date:   2023-03-12 15:16:52
# @Last Modified by:   lilz
# @Last Modified time: 2026-02-11 16:54:03


from pathlib import Path

from atomse.utils import group_seq


class VaspPotcar(object):

    def __init__(self, atoms, potentials=None):
        self.atoms = atoms
        self.potentials = potentials if potentials is not None else VaspPotcar.MP_POTENTIALS
        self.PAW_PBE_PATH = Path.home()/'POTCAR/PAW_PBE'

    def __str__(self):
        return self.generate_POTCAR()

    def generate_POTCAR(self):
        # 生成POTCAR
        str_ = ''
        elmt_types, elmt_cnts = group_seq(self.atoms.get_chemical_symbols())
        for elmt in elmt_types:
            with open(self.PAW_PBE_PATH/self.potentials[elmt]/'POTCAR', 'r') as f:
                str_ += f.read()
        return str_

    def write_POTCAR(self, outdir):
        # 写入POTCAR文件
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        POTCAR_path = outdir/'POTCAR'
        with open(POTCAR_path, 'w', newline='') as potcar:
            potcar.write(self.generate_POTCAR())
        return POTCAR_path

    def get_total_electrons(self):
        """
        @brief      Gets the total electrons, for modifing the NELECT of INCAR

        @return     The total electrons.
        """
        total_electrons = 0
        elmt_types, elmt_cnts = group_seq(self.atoms.get_chemical_symbols())
        for elmt_type, elmt_cnt in zip(elmt_types, elmt_cnts):
            total_electrons += (self.get_electrons(elmt_type)*elmt_cnt)
        return total_electrons

    def get_electrons(self, elmt):
        potcar_file = self.PAW_PBE_PATH/self.potentials[elmt]/'POTCAR'
        with open(potcar_file) as fd:
            cnt = 1
            while True:
                line = fd.readline()
                if cnt == 2:
                    break
                cnt += 1
            return int(line.split('.')[0].strip())

    MP_POTENTIALS = {
        'Ac': 'Ac',
        'Ag': 'Ag',
        'Al': 'Al',
        'Am': 'Am',
        'Ar': 'Ar',
        'As': 'As',
        'At': 'At',
        'Au': 'Au',
        'B': 'B',
        'Ba': 'Ba_sv',
        'Be': 'Be_sv',
        'Bi': 'Bi',
        'Br': 'Br',
        'C': 'C',
        'Ca': 'Ca_sv',
        'Cd': 'Cd',
        'Ce': 'Ce',
        'Cf': 'Cf',
        'Cl': 'Cl',
        'Cm': 'Cm',
        'Co': 'Co',
        'Cr': 'Cr_pv',
        'Cs': 'Cs_sv',
        'Cu': 'Cu_pv',
        'Dy': 'Dy_3',
        'Er': 'Er_3',
        'Eu': 'Eu',
        'F': 'F',
        'Fe': 'Fe_pv',
        'Fr': 'Fr_sv',
        'Ga': 'Ga_d',
        'Gd': 'Gd',
        'Ge': 'Ge_d',
        'H': 'H',
        'He': 'He',
        'Hf': 'Hf_pv',
        'Hg': 'Hg',
        'Ho': 'Ho_3',
        'I': 'I',
        'In': 'In_d',
        'Ir': 'Ir',
        'K': 'K_sv',
        'Kr': 'Kr',
        'La': 'La',
        'Li': 'Li_sv',
        'Lu': 'Lu_3',
        'Mg': 'Mg_pv',
        'Mn': 'Mn_pv',
        'Mo': 'Mo_pv',
        'N': 'N',
        'Na': 'Na_pv',
        'Nb': 'Nb_pv',
        'Nd': 'Nd_3',
        'Ne': 'Ne',
        'Ni': 'Ni_pv',
        'Np': 'Np',
        'O': 'O',
        'Os': 'Os_pv',
        'P': 'P',
        'Pa': 'Pa',
        'Pb': 'Pb_d',
        'Pd': 'Pd',
        'Pm': 'Pm_3',
        'Po': 'Po_d',
        'Pr': 'Pr_3',
        'Pt': 'Pt',
        'Pu': 'Pu',
        'Ra': 'Ra_sv',
        'Rb': 'Rb_sv',
        'Re': 'Re_pv',
        'Rh': 'Rh_pv',
        'Rn': 'Rn',
        'Ru': 'Ru_pv',
        'S': 'S',
        'Sb': 'Sb',
        'Sc': 'Sc_sv',
        'Se': 'Se',
        'Si': 'Si',
        'Sm': 'Sm_3',
        'Sn': 'Sn_d',
        'Sr': 'Sr_sv',
        'Ta': 'Ta_pv',
        'Tb': 'Tb_3',
        'Tc': 'Tc_pv',
        'Te': 'Te',
        'Th': 'Th',
        'Ti': 'Ti_pv',
        'Tl': 'Tl_d',
        'Tm': 'Tm_3',
        'U': 'U',
        'V': 'V_pv',
        'W': 'W_sv',
        'Xe': 'Xe',
        'Y': 'Y_sv',
        # 2023-05-02':'change Yb_2 to Yb_3 as Yb_2 gives incorrect thermodynamics for most systems with Yb3+'
        # https':'//github.com/materialsproject/pymatgen/issues/2968'
        'Yb': 'Yb_3',
        'Zn': 'Zn',
        'Zr': 'Zr_sv'
    }
