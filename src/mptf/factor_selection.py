# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 18:03:11 2026

@author: G_BRUNO
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence, Union, Tuple, Optional

import numpy as np
import pandas as pd


# ---------- 1) Define base factor groups (as tuples/lists), then convert to sets ----------

ANOM_FACTORS = (
    "ANOM_size", "ANOM_value", "ANOM_prof", "ANOM_valprof", "ANOM_fscore",
    "ANOM_debtiss", "ANOM_repurch", "ANOM_nissa", "ANOM_accruals", "ANOM_growth",
    "ANOM_aturnover", "ANOM_gmargins", "ANOM_divp", "ANOM_ep", "ANOM_cfp",
    "ANOM_noa", "ANOM_inv", "ANOM_invcap", "ANOM_igrowth", "ANOM_sgrowth",
    "ANOM_lev", "ANOM_roaa", "ANOM_roea", "ANOM_sp", "ANOM_gltnoa",
    "ANOM_mom", "ANOM_indmom", "ANOM_valmom", "ANOM_valmomprof", "ANOM_shortint",
    "ANOM_mom12", "ANOM_momrev", "ANOM_lrrev", "ANOM_valuem", "ANOM_nissm",
    "ANOM_sue", "ANOM_roe", "ANOM_rome", "ANOM_roa", "ANOM_strev",
    "ANOM_ivol", "ANOM_betaarb", "ANOM_season", "ANOM_indrrev",
    "ANOM_indrrevlv", "ANOM_indmomrev", "ANOM_ciss", "ANOM_price",
    "ANOM_age", "ANOM_shvol",
)

FR_FACTORS = (
    "FR_pe_exi", "FR_pe_inc", "FR_ps", "FR_pcf", "FR_evm", "FR_bm", "FR_capei",
    "FR_dpr", "FR_npm", "FR_opmbd", "FR_opmad", "FR_gpm", "FR_ptpm", "FR_cfm",
    "FR_roa", "FR_roe", "FR_roce", "FR_aftret_eq", "FR_aftret_invcapx",
    "FR_aftret_equity", "FR_pretret_noa", "FR_pretret_earnat",
    "FR_equity_invcap", "FR_debt_invcap", "FR_totdebt_invcap", "FR_int_debt",
    "FR_int_totdebt", "FR_cash_lt", "FR_invt_act", "FR_rect_act", "FR_debt_at",
    "FR_short_debt", "FR_curr_debt", "FR_lt_debt", "FR_fcf_ocf", "FR_adv_sale",
    "FR_profit_lct", "FR_debt_ebitda", "FR_ocf_lct", "FR_lt_ppent",
    "FR_dltt_be", "FR_debt_assets", "FR_debt_capital", "FR_de_ratio",
    "FR_intcov", "FR_cash_ratio", "FR_quick_ratio", "FR_curr_ratio",
    "FR_capital_ratio", "FR_cash_debt", "FR_inv_turn", "FR_at_turn",
    "FR_rect_turn", "FR_pay_turn", "FR_sale_invcap", "FR_sale_equity",
    "FR_sale_nwc", "FR_rd_sale", "FR_accrual", "FR_gprof", "FR_be",
    "FR_cash_conversion", "FR_efftax", "FR_intcov_ratio", "FR_staff_sale",
    "FR_divyield", "FR_ptb", "FR_peg_trailing",
    "FR_retminus1m", "FR_retminus2m", "FR_retminus3m", "FR_retminus4m",
    "FR_retminus5m", "FR_retminus6m", "FR_retminus7m", "FR_retminus8m",
    "FR_retminus9m", "FR_retminus10m", "FR_retminus11m", "FR_retminus12m",
)

FF6_FACTORS = ("FF6_MktMinusRF", "FF6_SMB", "FF6_HML", "FF6_RMW", "FF6_CMA", "FF6_Mom")
CSFF6_FACTORS = ("FF6_SMB", "FF6_HML", "FF6_RMW", "FF6_CMA", "FF6_Mom")

HXZ_FACTORS = ("HXZ_MktMinusRF", "HXZ_R_ME", "HXZ_R_IA", "HXZ_R_ROE", "HXZ_R_EG")
CSHXZ_FACTORS = ("HXZ_R_ME", "HXZ_R_IA", "HXZ_R_ROE", "HXZ_R_EG")

MKT_FACTORS = ("FF6_MktMinusRF",)

KNSSPR_FACTORS = ("ANOM_size", "ANOM_value", "ANOM_prof", "ANOM_growth", "ANOM_mom")

# --- GFD / OAP subsets (copy-pasted from your MATLAB) ---
GFDALL_FACTORS = (
    "GFD_age", "GFD_aliq_at", "GFD_aliq_mat", "GFD_ami_126d", "GFD_at_be",
    "GFD_at_gr1", "GFD_at_me", "GFD_at_turnover", "GFD_be_gr1a", "GFD_be_me",
    "GFD_beta_60m", "GFD_beta_dimson_21d", "GFD_betabab_1260d", "GFD_betadown_252d",
    "GFD_bev_mev", "GFD_bidaskhl_21d", "GFD_capex_abn", "GFD_capx_gr1",
    "GFD_capx_gr2", "GFD_capx_gr3", "GFD_cash_at", "GFD_chcsho_12m",
    "GFD_coa_gr1a", "GFD_col_gr1a", "GFD_cop_at", "GFD_cop_atl1",
    "GFD_corr_1260d", "GFD_coskew_21d", "GFD_cowc_gr1a", "GFD_dbnetis_at",
    "GFD_debt_gr3", "GFD_debt_me", "GFD_dgp_dsale", "GFD_div12m_me",
    "GFD_dolvol_126d", "GFD_dolvol_var_126d", "GFD_dsale_dinv", "GFD_dsale_drec",
    "GFD_dsale_dsga", "GFD_earnings_variability", "GFD_ebit_bev", "GFD_ebit_sale",
    "GFD_ebitda_mev", "GFD_emp_gr1", "GFD_eq_dur", "GFD_eqnetis_at",
    "GFD_eqnpo_12m", "GFD_eqnpo_me", "GFD_eqpo_me", "GFD_f_score",
    "GFD_fcf_me", "GFD_fnl_gr1a", "GFD_gp_at", "GFD_gp_atl1",
    "GFD_inv_gr1", "GFD_inv_gr1a", "GFD_iskew_capm_21d", "GFD_iskew_ff3_21d",
    "GFD_iskew_hxz4_21d", "GFD_ival_me", "GFD_ivol_capm_21d", "GFD_ivol_capm_252d",
    "GFD_ivol_ff3_21d", "GFD_ivol_hxz4_21d", "GFD_kz_index", "GFD_lnoa_gr1a",
    "GFD_lti_gr1a", "GFD_market_equity", "GFD_mispricing_mgmt", "GFD_mispricing_perf",
    "GFD_ncoa_gr1a", "GFD_ncol_gr1a", "GFD_netdebt_me", "GFD_netis_at",
    "GFD_nfna_gr1a", "GFD_ni_ar1", "GFD_ni_be", "GFD_ni_inc8q", "GFD_ni_ivol",
    "GFD_ni_me", "GFD_niq_at", "GFD_niq_at_chg1", "GFD_niq_be", "GFD_niq_be_chg1",
    "GFD_niq_su", "GFD_nncoa_gr1a", "GFD_noa_at", "GFD_noa_gr1a", "GFD_o_score",
    "GFD_oaccruals_at", "GFD_oaccruals_ni", "GFD_ocf_at", "GFD_ocf_at_chg1",
    "GFD_ocf_me", "GFD_ocfq_saleq_std", "GFD_op_at", "GFD_op_atl1", "GFD_ope_be",
    "GFD_ope_bel1", "GFD_opex_at", "GFD_pi_nix", "GFD_ppeinv_gr1a", "GFD_prc",
    "GFD_prc_highprc_252d", "GFD_qmj", "GFD_qmj_growth", "GFD_qmj_prof",
    "GFD_qmj_safety", "GFD_rd5_at", "GFD_rd_me", "GFD_rd_sale", "GFD_resff3_12_1",
    "GFD_resff3_6_1", "GFD_ret_12_1", "GFD_ret_12_7", "GFD_ret_1_0", "GFD_ret_3_1",
    "GFD_ret_60_12", "GFD_ret_6_1", "GFD_ret_9_1", "GFD_rmax1_21d", "GFD_rmax5_21d",
    "GFD_rmax5_rvol_21d", "GFD_rskew_21d", "GFD_rvol_21d", "GFD_sale_bev",
    "GFD_sale_emp_gr1", "GFD_sale_gr1", "GFD_sale_gr3", "GFD_sale_me", "GFD_saleq_gr1",
    "GFD_saleq_su", "GFD_seas_11_15an", "GFD_seas_11_15na", "GFD_seas_16_20an",
    "GFD_seas_16_20na", "GFD_seas_1_1an", "GFD_seas_1_1na", "GFD_seas_2_5an",
    "GFD_seas_2_5na", "GFD_seas_6_10an", "GFD_seas_6_10na", "GFD_sti_gr1a",
    "GFD_taccruals_at", "GFD_taccruals_ni", "GFD_tangibility", "GFD_tax_gr1a",
    "GFD_turnover_126d", "GFD_turnover_var_126d", "GFD_z_score", "GFD_zero_trades_126d",
    "GFD_zero_trades_21d", "GFD_zero_trades_252d",
)

GFDSparseFF_FACTORS = ("GFD_at_gr1", "GFD_be_me", "GFD_ope_be", "GFD_market_equity", "GFD_ret_6_1")
GFDSparse_QfactorsFF = ("GFD_at_gr1", "GFD_ope_be")     # {1},{2} in MATLAB
GFDSparse_QfactorsHXZ = ("GFD_at_gr1", "GFD_niq_be")    # {1},{2} in MATLAB

GFDPROF_QFACTORS = (
    "GFD_cop_at","GFD_cop_atl1","GFD_dgp_dsale","GFD_ebit_bev","GFD_ebit_sale",
    "GFD_f_score","GFD_gp_at","GFD_gp_atl1","GFD_ni_be","GFD_ni_inc8q","GFD_niq_at",
    "GFD_niq_be","GFD_o_score","GFD_ocf_at","GFD_op_at","GFD_op_atl1","GFD_ope_be",
    "GFD_ope_bel1","GFD_qmj_prof",
)
GFDINV_QFACTORS = (
    "GFD_at_gr1","GFD_be_gr1a","GFD_capex_abn","GFD_capx_gr1","GFD_capx_gr2","GFD_capx_gr3",
    "GFD_coa_gr1a","GFD_col_gr1a","GFD_cowc_gr1a","GFD_debt_gr3","GFD_emp_gr1","GFD_fnl_gr1a",
    "GFD_inv_gr1","GFD_inv_gr1a","GFD_lnoa_gr1a","GFD_mispricing_mgmt","GFD_ncoa_gr1a",
    "GFD_nncoa_gr1a","GFD_noa_gr1a","GFD_ppeinv_gr1a",
)

OAP_QFACTORS = (
    "OAP_AssetGrowth","OAP_ChEQ","OAP_DelEqu","OAP_DelLTI","OAP_dNoa","OAP_GrLTNOA",
    "OAP_InvGrowth","OAP_Investment","OAP_InvestPPEInv","OAP_CBOperProf","OAP_FEPS","OAP_GP",
    "OAP_OperProf","OAP_OperProfRD","OAP_roaq","OAP_RoE",
)
OAP_PROF = ("OAP_CBOperProf","OAP_FEPS","OAP_GP","OAP_OperProf","OAP_OperProfRD","OAP_roaq","OAP_RoE")
OAP_INV = ("OAP_AssetGrowth","OAP_ChEQ","OAP_DelEqu","OAP_DelLTI","OAP_dNoa","OAP_GrLTNOA","OAP_InvGrowth","OAP_Investment","OAP_InvestPPEInv")

OAP_FFQ = ("OAP_AssetGrowth", "OAP_OperProf")  # {1},{2} in MATLAB
OAP_HXZQ = ("OAP_AssetGrowth", "OAP_RoE")      # {1},{2} in MATLAB


# ---------- 2) Selection mapping ----------
# Store sets, and build composites via union.

BASE_SETS = {
    "ANOM": set(ANOM_FACTORS),
    "FR": set(FR_FACTORS),
    "FF6": set(FF6_FACTORS),
    "CSFF6": set(CSFF6_FACTORS),
    "HXZ": set(HXZ_FACTORS),
    "CSHXZ": set(CSHXZ_FACTORS),
    "MKT": set(MKT_FACTORS),
    "KNSSPR": set(KNSSPR_FACTORS),

    "GFDALL": set(GFDALL_FACTORS),
    "GFDSPRFF": set(GFDSparseFF_FACTORS),

    "GFDSPRQFF": set(GFDSparse_QfactorsFF),
    "GFDSPRQHXZ": set(GFDSparse_QfactorsHXZ),

    "GFDPROFAIQ": set(GFDPROF_QFACTORS),
    "GFDINVAIQ": set(GFDINV_QFACTORS),

    "OAPAIQ": set(OAP_QFACTORS),
    "OAPPROFAIQ": set(OAP_PROF),
    "OAPINVAIQ": set(OAP_INV),

    "OAPSPRQFF": set(OAP_FFQ),
    "OAPSPRQHXZ": set(OAP_HXZQ),
}

# Composites / “single-element picks” that were gfdsparse_QfactorsFF{1}/{2} etc.
# (This matches your MATLAB intent, but avoids MATLAB's ismember-with-char weirdness.)
DERIVED_SETS = {
    "FIN_KNS": lambda: BASE_SETS["ANOM"] | BASE_SETS["FR"],

    "GFDAIQ": lambda: BASE_SETS["GFDPROFAIQ"] | BASE_SETS["GFDINVAIQ"],

    # single-factor picks (MATLAB {...}{1}/{2})
    "GFDINVFF": lambda: {GFDSparse_QfactorsFF[0]},
    "GFDPROFFF": lambda: {GFDSparse_QfactorsFF[1]},
    "GFDINVHXZ": lambda: {GFDSparse_QfactorsHXZ[0]},
    "GFDPROFHXZ": lambda: {GFDSparse_QfactorsHXZ[1]},

    "OAPINVFF": lambda: {OAP_FFQ[0]},
    "OAPPROFFF": lambda: {OAP_FFQ[1]},
    "OAPINVHXZ": lambda: {OAP_HXZQ[0]},
    "OAPPROFHXZ": lambda: {OAP_HXZQ[1]},
}


@lru_cache(maxsize=None)
def _resolve_selection(selection_id: str) -> set[str]:
    key = selection_id.upper()
    if key in BASE_SETS:
        return BASE_SETS[key]
    if key in DERIVED_SETS:
        return set(DERIVED_SETS[key]())
    raise ValueError(f"Invalid selection_id: {selection_id!r}")


def factorselection(
    factor_id: Union[pd.Index, pd.Series, Sequence[str], np.ndarray],
    selection_id: str,
    *,
    return_selected: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list[str]]]:
    """
    Python analogue of MATLAB factorselection.

    Parameters
    ----------
    factor_id : list/array/Index of factor names (typically your parquet columns)
    selection_id : string key (case-insensitive), e.g. 'ANOM', 'GFDALL', 'GFDAIQ'
    return_selected : if True, also return the selected names

    Returns
    -------
    mask : np.ndarray[bool] aligned with factor_id
    selected : list[str] (optional)
    """
    sel_set = _resolve_selection(selection_id)

    # Normalize factor_id to an Index for nice handling
    idx = pd.Index(factor_id) if not isinstance(factor_id, pd.Index) else factor_id

    # Vectorized membership test
    mask = np.asarray(idx.isin(sel_set), dtype=bool)   # <-- NO .to_numpy()

    if return_selected:
        selected = idx[mask].tolist()
        return mask, selected

    return mask