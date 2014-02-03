#include "head.h"

void eval_spec_rates (const Real * fwd_rates, const Real * rev_rates, const Real * pres_mod, Real * sp_rates) {
  sp_rates[0] = -(fwd_rates[0] - rev_rates[0]) + (fwd_rates[1] - rev_rates[1]) + (fwd_rates[2] - rev_rates[2])
               + (fwd_rates[3] - rev_rates[3]) + 2.0 * (fwd_rates[5] - rev_rates[5]) * pres_mod[0]
               + 2.0 * (fwd_rates[6] - rev_rates[6]) + 2.0 * (fwd_rates[7] - rev_rates[7])
               - (fwd_rates[11] - rev_rates[11]) * pres_mod[2] + (fwd_rates[12] - rev_rates[12]) * pres_mod[3]
               + (fwd_rates[13] - rev_rates[13]) - (fwd_rates[14] - rev_rates[14]) * pres_mod[4]
               - (fwd_rates[15] - rev_rates[15]) - (fwd_rates[16] - rev_rates[16]) - (fwd_rates[22] - rev_rates[22])
               - (fwd_rates[23] - rev_rates[23]);

  sp_rates[1] = -(fwd_rates[1] - rev_rates[1]) - (fwd_rates[2] - rev_rates[2]) - (fwd_rates[3] - rev_rates[3])
               - (fwd_rates[5] - rev_rates[5]) * pres_mod[0] - (fwd_rates[6] - rev_rates[6])
               - (fwd_rates[7] - rev_rates[7]) + (fwd_rates[15] - rev_rates[15]) + (fwd_rates[23] - rev_rates[23]);

  sp_rates[2] = (fwd_rates[0] - rev_rates[0]) - (fwd_rates[1] - rev_rates[1]) - (fwd_rates[2] - rev_rates[2])
               + (fwd_rates[4] - rev_rates[4]) - 2.0 * (fwd_rates[8] - rev_rates[8]) * pres_mod[1]
               - 2.0 * (fwd_rates[9] - rev_rates[9]) - 2.0 * (fwd_rates[10] - rev_rates[10])
               - (fwd_rates[11] - rev_rates[11]) * pres_mod[2] - (fwd_rates[17] - rev_rates[17])
               - (fwd_rates[24] - rev_rates[24]);

  sp_rates[3] = (fwd_rates[0] - rev_rates[0]) + (fwd_rates[1] - rev_rates[1]) + (fwd_rates[2] - rev_rates[2])
               - (fwd_rates[3] - rev_rates[3]) - 2.0 * (fwd_rates[4] - rev_rates[4]) + (fwd_rates[11] - rev_rates[11]) * pres_mod[2]
               + (fwd_rates[12] - rev_rates[12]) * pres_mod[3] + (fwd_rates[13] - rev_rates[13])
               + 2.0 * (fwd_rates[16] - rev_rates[16]) + (fwd_rates[17] - rev_rates[17])
               - (fwd_rates[18] - rev_rates[18]) + 2.0 * (fwd_rates[21] - rev_rates[21]) * pres_mod[5]
               + (fwd_rates[22] - rev_rates[22]) + (fwd_rates[24] - rev_rates[24]) - (fwd_rates[25] - rev_rates[25])
               - (fwd_rates[26] - rev_rates[26]);

  sp_rates[4] = (fwd_rates[3] - rev_rates[3]) + (fwd_rates[4] - rev_rates[4]) - (fwd_rates[12] - rev_rates[12]) * pres_mod[3]
               - (fwd_rates[13] - rev_rates[13]) + (fwd_rates[18] - rev_rates[18]) + (fwd_rates[22] - rev_rates[22])
               + (fwd_rates[25] - rev_rates[25]) + (fwd_rates[26] - rev_rates[26]);

  sp_rates[5] = -(fwd_rates[0] - rev_rates[0]) + (fwd_rates[8] - rev_rates[8]) * pres_mod[1]
               + (fwd_rates[9] - rev_rates[9]) + (fwd_rates[10] - rev_rates[10]) - (fwd_rates[14] - rev_rates[14]) * pres_mod[4]
               + (fwd_rates[15] - rev_rates[15]) + (fwd_rates[17] - rev_rates[17]) + (fwd_rates[18] - rev_rates[18])
               + (fwd_rates[19] - rev_rates[19]) + (fwd_rates[20] - rev_rates[20]);

  sp_rates[6] = (fwd_rates[14] - rev_rates[14]) * pres_mod[4] - (fwd_rates[15] - rev_rates[15])
               - (fwd_rates[16] - rev_rates[16]) - (fwd_rates[17] - rev_rates[17]) - (fwd_rates[18] - rev_rates[18])
               - 2.0 * (fwd_rates[19] - rev_rates[19]) - 2.0 * (fwd_rates[20] - rev_rates[20])
               + (fwd_rates[23] - rev_rates[23]) + (fwd_rates[24] - rev_rates[24]) + (fwd_rates[25] - rev_rates[25])
               + (fwd_rates[26] - rev_rates[26]);

  sp_rates[7] = (fwd_rates[19] - rev_rates[19]) + (fwd_rates[20] - rev_rates[20]) - (fwd_rates[21] - rev_rates[21]) * pres_mod[5]
               - (fwd_rates[22] - rev_rates[22]) - (fwd_rates[23] - rev_rates[23]) - (fwd_rates[24] - rev_rates[24])
               - (fwd_rates[25] - rev_rates[25]) - (fwd_rates[26] - rev_rates[26]);

  sp_rates[8] = 0.0;

  sp_rates[9] = 0.0;

  sp_rates[10] = 0.0;

  sp_rates[11] = 0.0;

  sp_rates[12] = 0.0;

} // end eval_spec_rates

