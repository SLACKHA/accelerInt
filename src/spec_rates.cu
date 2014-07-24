#include "header.h"

__device__ void eval_spec_rates (const Real * fwd_rates, const Real * pres_mod, Real * sp_rates) {
  sp_rates[0] = -fwd_rates[0] + fwd_rates[1] + fwd_rates[2] - fwd_rates[3] + fwd_rates[4]
              - fwd_rates[5] + fwd_rates[6] - fwd_rates[7] + 2.0 * fwd_rates[10] * pres_mod[0]
              - 2.0 * fwd_rates[11] * pres_mod[1] + 2.0 * fwd_rates[12] - 2.0 * fwd_rates[13]
              + 2.0 * fwd_rates[14] - 2.0 * fwd_rates[15] - fwd_rates[22] * pres_mod[4]
              + fwd_rates[23] * pres_mod[5] + fwd_rates[24] * pres_mod[6] - fwd_rates[25] * pres_mod[7]
              + fwd_rates[26] - fwd_rates[27] - fwd_rates[28] * pres_mod[8] + fwd_rates[29] * pres_mod[9]
              - fwd_rates[30] + fwd_rates[31] - fwd_rates[32] + fwd_rates[33] - fwd_rates[44]
              + fwd_rates[45] - fwd_rates[46] + fwd_rates[47];

  sp_rates[1] = -fwd_rates[2] + fwd_rates[3] - fwd_rates[4] + fwd_rates[5] - fwd_rates[6]
              + fwd_rates[7] - fwd_rates[10] * pres_mod[0] + fwd_rates[11] * pres_mod[1]
              - fwd_rates[12] + fwd_rates[13] - fwd_rates[14] + fwd_rates[15] + fwd_rates[30]
              - fwd_rates[31] + fwd_rates[46] - fwd_rates[47];

  sp_rates[2] = fwd_rates[0] - fwd_rates[1] - fwd_rates[2] + fwd_rates[3] - fwd_rates[4]
              + fwd_rates[5] + fwd_rates[8] - fwd_rates[9] - 2.0 * fwd_rates[16] * pres_mod[2]
              + 2.0 * fwd_rates[17] * pres_mod[3] - 2.0 * fwd_rates[18] + 2.0 * fwd_rates[19]
              - 2.0 * fwd_rates[20] + 2.0 * fwd_rates[21] - fwd_rates[22] * pres_mod[4]
              + fwd_rates[23] * pres_mod[5] - fwd_rates[34] + fwd_rates[35] - fwd_rates[48]
              + fwd_rates[49];

  sp_rates[3] = fwd_rates[0] - fwd_rates[1] + fwd_rates[2] - fwd_rates[3] + fwd_rates[4]
              - fwd_rates[5] - fwd_rates[6] + fwd_rates[7] - 2.0 * fwd_rates[8] + 2.0 * fwd_rates[9]
              + fwd_rates[22] * pres_mod[4] - fwd_rates[23] * pres_mod[5] + fwd_rates[24] * pres_mod[6]
              - fwd_rates[25] * pres_mod[7] + fwd_rates[26] - fwd_rates[27] + 2.0 * fwd_rates[32]
              - 2.0 * fwd_rates[33] + fwd_rates[34] - fwd_rates[35] - fwd_rates[36] + fwd_rates[37]
              + 2.0 * fwd_rates[42] * pres_mod[10] - 2.0 * fwd_rates[43] * pres_mod[11]
              + fwd_rates[44] - fwd_rates[45] + fwd_rates[48] - fwd_rates[49] - fwd_rates[50]
              + fwd_rates[51] - fwd_rates[52] + fwd_rates[53];

  sp_rates[4] = fwd_rates[6] - fwd_rates[7] + fwd_rates[8] - fwd_rates[9] - fwd_rates[24] * pres_mod[6]
              + fwd_rates[25] * pres_mod[7] - fwd_rates[26] + fwd_rates[27] + fwd_rates[36]
              - fwd_rates[37] + fwd_rates[44] - fwd_rates[45] + fwd_rates[50] - fwd_rates[51]
              + fwd_rates[52] - fwd_rates[53];

  sp_rates[5] = -fwd_rates[0] + fwd_rates[1] + fwd_rates[16] * pres_mod[2] - fwd_rates[17] * pres_mod[3]
              + fwd_rates[18] - fwd_rates[19] + fwd_rates[20] - fwd_rates[21] - fwd_rates[28] * pres_mod[8]
              + fwd_rates[29] * pres_mod[9] + fwd_rates[30] - fwd_rates[31] + fwd_rates[34]
              - fwd_rates[35] + fwd_rates[36] - fwd_rates[37] + fwd_rates[38] - fwd_rates[39]
              + fwd_rates[40] - fwd_rates[41];

  sp_rates[6] = fwd_rates[28] * pres_mod[8] - fwd_rates[29] * pres_mod[9] - fwd_rates[30]
              + fwd_rates[31] - fwd_rates[32] + fwd_rates[33] - fwd_rates[34] + fwd_rates[35]
              - fwd_rates[36] + fwd_rates[37] - 2.0 * fwd_rates[38] + 2.0 * fwd_rates[39]
              - 2.0 * fwd_rates[40] + 2.0 * fwd_rates[41] + fwd_rates[46] - fwd_rates[47]
              + fwd_rates[48] - fwd_rates[49] + fwd_rates[50] - fwd_rates[51] + fwd_rates[52]
              - fwd_rates[53];

  sp_rates[7] = fwd_rates[38] - fwd_rates[39] + fwd_rates[40] - fwd_rates[41] - fwd_rates[42] * pres_mod[10]
              + fwd_rates[43] * pres_mod[11] - fwd_rates[44] + fwd_rates[45] - fwd_rates[46]
              + fwd_rates[47] - fwd_rates[48] + fwd_rates[49] - fwd_rates[50] + fwd_rates[51]
              - fwd_rates[52] + fwd_rates[53];

  sp_rates[8] = 0.0;

  sp_rates[9] = 0.0;

  sp_rates[10] = 0.0;

  sp_rates[11] = 0.0;

  sp_rates[12] = 0.0;

} // end eval_spec_rates

