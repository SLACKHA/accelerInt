#include "header.h"

void eval_spec_rates (const Real * fwd_rates, const Real * rev_rates, const Real * pres_mod, Real * sp_rates) {
  sp_rates[0] = -(fwd_rates[2] - rev_rates[2]) + (fwd_rates[7] - rev_rates[7]) + (fwd_rates[38] - rev_rates[38]) * pres_mod[4]
              + (fwd_rates[39] - rev_rates[39]) + (fwd_rates[40] - rev_rates[40]) + (fwd_rates[41] - rev_rates[41])
              + (fwd_rates[44] - rev_rates[44]) + (fwd_rates[46] - rev_rates[46]) + (fwd_rates[48] - rev_rates[48])
              + (fwd_rates[50] - rev_rates[50]) + (fwd_rates[52] - rev_rates[52]) + (fwd_rates[54] - rev_rates[54])
              + (fwd_rates[57] - rev_rates[57]) + (fwd_rates[59] - rev_rates[59]) + (fwd_rates[64] - rev_rates[64])
              + (fwd_rates[67] - rev_rates[67]) + (fwd_rates[68] - rev_rates[68]) + (fwd_rates[72] - rev_rates[72])
              + (fwd_rates[74] - rev_rates[74]) + (fwd_rates[76] - rev_rates[76]) + (fwd_rates[77] - rev_rates[77])
              + (fwd_rates[79] - rev_rates[79]) - (fwd_rates[82] - rev_rates[82]) * pres_mod[18]
              - (fwd_rates[83] - rev_rates[83]) - (fwd_rates[125] - rev_rates[125]) - (fwd_rates[135] - rev_rates[134])
              + (fwd_rates[136] - rev_rates[135]) - (fwd_rates[145] - rev_rates[144]) - (fwd_rates[171] - rev_rates[170])
              + (fwd_rates[173] - rev_rates[172]) * pres_mod[26] + (fwd_rates[190] - rev_rates[189])
              + (fwd_rates[196] - rev_rates[195]) + (fwd_rates[201] - rev_rates[200]) + (fwd_rates[208] - rev_rates[207])
              + (fwd_rates[213] - rev_rates[212]) - (fwd_rates[220] - rev_rates[219]) + (fwd_rates[265] - rev_rates[264])
              + (fwd_rates[275] - rev_rates[274]) + (fwd_rates[276] - rev_rates[275]) + fwd_rates[283]
              + fwd_rates[287] - (fwd_rates[288] - rev_rates[285]) * pres_mod[36] + fwd_rates[292]
              + (fwd_rates[298] - rev_rates[290]) + fwd_rates[299] + (fwd_rates[308] - rev_rates[293])
              + (fwd_rates[313] - rev_rates[298]);

  sp_rates[1] = -(fwd_rates[1] - rev_rates[1]) * pres_mod[1] + (fwd_rates[2] - rev_rates[2])
              + (fwd_rates[5] - rev_rates[5]) + (fwd_rates[6] - rev_rates[6]) + (fwd_rates[8] - rev_rates[8])
              + (fwd_rates[9] - rev_rates[9]) + (fwd_rates[13] - rev_rates[13]) + (fwd_rates[20] - rev_rates[20])
              + (fwd_rates[23] - rev_rates[23]) + (fwd_rates[27] - rev_rates[27]) - (fwd_rates[32] - rev_rates[32]) * pres_mod[3]
              - (fwd_rates[33] - rev_rates[33]) - (fwd_rates[34] - rev_rates[34]) - (fwd_rates[35] - rev_rates[35])
              - (fwd_rates[36] - rev_rates[36]) - (fwd_rates[37] - rev_rates[37]) - 2.0 * (fwd_rates[38] - rev_rates[38]) * pres_mod[4]
              - 2.0 * (fwd_rates[39] - rev_rates[39]) - 2.0 * (fwd_rates[40] - rev_rates[40])
              - 2.0 * (fwd_rates[41] - rev_rates[41]) - (fwd_rates[42] - rev_rates[42]) * pres_mod[5]
              - (fwd_rates[43] - rev_rates[43]) - (fwd_rates[44] - rev_rates[44]) - (fwd_rates[45] - rev_rates[45])
              - (fwd_rates[46] - rev_rates[46]) - (fwd_rates[47] - rev_rates[47]) - (fwd_rates[48] - rev_rates[48])
              - (fwd_rates[49] - rev_rates[49]) * pres_mod[6] - (fwd_rates[50] - rev_rates[50])
              - (fwd_rates[51] - rev_rates[51]) * pres_mod[7] - (fwd_rates[52] - rev_rates[52])
              - (fwd_rates[53] - rev_rates[53]) * pres_mod[8] - (fwd_rates[54] - rev_rates[54])
              - (fwd_rates[55] - rev_rates[55]) * pres_mod[9] - (fwd_rates[56] - rev_rates[56]) * pres_mod[10]
              - (fwd_rates[57] - rev_rates[57]) - (fwd_rates[58] - rev_rates[58]) * pres_mod[11]
              - (fwd_rates[59] - rev_rates[59]) - (fwd_rates[60] - rev_rates[60]) - (fwd_rates[61] - rev_rates[61])
              - (fwd_rates[62] - rev_rates[62]) * pres_mod[12] - (fwd_rates[64] - rev_rates[64])
              - (fwd_rates[65] - rev_rates[65]) - (fwd_rates[66] - rev_rates[66]) - (fwd_rates[67] - rev_rates[67])
              - (fwd_rates[68] - rev_rates[68]) - (fwd_rates[69] - rev_rates[69]) * pres_mod[13]
              - (fwd_rates[70] - rev_rates[70]) * pres_mod[14] - (fwd_rates[71] - rev_rates[71]) * pres_mod[15]
              - (fwd_rates[72] - rev_rates[72]) - (fwd_rates[73] - rev_rates[73]) * pres_mod[16]
              - (fwd_rates[74] - rev_rates[74]) - (fwd_rates[75] - rev_rates[75]) * pres_mod[17]
              - (fwd_rates[76] - rev_rates[76]) - (fwd_rates[77] - rev_rates[77]) - (fwd_rates[78] - rev_rates[78])
              - (fwd_rates[79] - rev_rates[79]) - (fwd_rates[80] - rev_rates[80]) + (fwd_rates[83] - rev_rates[83])
              + (fwd_rates[89] - rev_rates[89]) + (fwd_rates[90] - rev_rates[90]) + (fwd_rates[91] - rev_rates[91])
              + (fwd_rates[93] - rev_rates[93]) + (fwd_rates[98] - rev_rates[98]) + (fwd_rates[105] - rev_rates[105])
              + (fwd_rates[106] - rev_rates[106]) + (fwd_rates[107] - rev_rates[107]) + (fwd_rates[122] - rev_rates[122])
              + (fwd_rates[123] - rev_rates[123]) + (fwd_rates[125] - rev_rates[125]) + (fwd_rates[126] - rev_rates[126])
              + (fwd_rates[127] - rev_rates[127]) + (fwd_rates[128] - rev_rates[128]) + (fwd_rates[129] - rev_rates[129])
              + (fwd_rates[132] - rev_rates[132]) + fwd_rates[134] + (fwd_rates[135] - rev_rates[134])
              + (fwd_rates[137] - rev_rates[136]) + (fwd_rates[143] - rev_rates[142]) + (fwd_rates[145] - rev_rates[144])
              + (fwd_rates[148] - rev_rates[147]) + (fwd_rates[158] - rev_rates[157]) + (fwd_rates[165] - rev_rates[164])
              + (fwd_rates[166] - rev_rates[165]) * pres_mod[25] + (fwd_rates[171] - rev_rates[170])
              + (fwd_rates[179] - rev_rates[178]) - (fwd_rates[182] - rev_rates[181]) - (fwd_rates[188] - rev_rates[187])
              + (fwd_rates[189] - rev_rates[188]) - (fwd_rates[190] - rev_rates[189]) + (fwd_rates[191] - rev_rates[190])
              + (fwd_rates[195] - rev_rates[194]) + (fwd_rates[198] - rev_rates[197]) + (fwd_rates[200] - rev_rates[199])
              - (fwd_rates[201] - rev_rates[200]) + (fwd_rates[203] - rev_rates[202]) + (fwd_rates[204] - rev_rates[203]) * pres_mod[29]
              - (fwd_rates[208] - rev_rates[207]) - (fwd_rates[211] - rev_rates[210]) * pres_mod[30]
              - (fwd_rates[213] - rev_rates[212]) + (fwd_rates[217] - rev_rates[216]) + (fwd_rates[220] - rev_rates[219])
              - (fwd_rates[222] - rev_rates[221]) + (fwd_rates[223] - rev_rates[222]) + (fwd_rates[229] - rev_rates[228]) * pres_mod[32]
              + (fwd_rates[230] - rev_rates[229]) + (fwd_rates[233] - rev_rates[232]) + (fwd_rates[234] - rev_rates[233])
              - (fwd_rates[236] - rev_rates[235]) * pres_mod[33] + (fwd_rates[246] - rev_rates[245])
              + (fwd_rates[248] - rev_rates[247]) + (fwd_rates[250] - rev_rates[249]) + (fwd_rates[251] - rev_rates[250])
              + (fwd_rates[253] - rev_rates[252]) + (fwd_rates[256] - rev_rates[255]) + (fwd_rates[259] - rev_rates[258])
              - (fwd_rates[260] - rev_rates[259]) - (fwd_rates[264] - rev_rates[263]) - (fwd_rates[265] - rev_rates[264])
              - (fwd_rates[270] - rev_rates[269]) - (fwd_rates[271] - rev_rates[270]) + (fwd_rates[274] - rev_rates[273])
              - (fwd_rates[276] - rev_rates[275]) + fwd_rates[283] + (fwd_rates[284] - rev_rates[282])
              + (fwd_rates[285] - rev_rates[283]) + 2.0 * fwd_rates[289] + 2.0 * fwd_rates[291]
              - (fwd_rates[298] - rev_rates[290]) - fwd_rates[299] - (fwd_rates[303] - rev_rates[291]) * pres_mod[37]
              + fwd_rates[304] - (fwd_rates[307] - rev_rates[292]) - (fwd_rates[308] - rev_rates[293])
              - (fwd_rates[313] - rev_rates[298]) - (fwd_rates[319] - rev_rates[304]) * pres_mod[40]
              - (fwd_rates[320] - rev_rates[305]);

  sp_rates[2] = -2.0 * (fwd_rates[0] - rev_rates[0]) * pres_mod[0] - (fwd_rates[1] - rev_rates[1]) * pres_mod[1]
              - (fwd_rates[2] - rev_rates[2]) - (fwd_rates[3] - rev_rates[3]) - (fwd_rates[4] - rev_rates[4])
              - (fwd_rates[5] - rev_rates[5]) - (fwd_rates[6] - rev_rates[6]) - (fwd_rates[7] - rev_rates[7])
              - (fwd_rates[8] - rev_rates[8]) - (fwd_rates[9] - rev_rates[9]) - (fwd_rates[10] - rev_rates[10])
              - (fwd_rates[11] - rev_rates[11]) * pres_mod[2] - (fwd_rates[12] - rev_rates[12])
              - (fwd_rates[13] - rev_rates[13]) - (fwd_rates[14] - rev_rates[14]) - (fwd_rates[15] - rev_rates[15])
              - (fwd_rates[16] - rev_rates[16]) - (fwd_rates[17] - rev_rates[17]) - (fwd_rates[18] - rev_rates[18])
              - (fwd_rates[19] - rev_rates[19]) - (fwd_rates[20] - rev_rates[20]) - (fwd_rates[21] - rev_rates[21])
              - (fwd_rates[22] - rev_rates[22]) - (fwd_rates[23] - rev_rates[23]) - (fwd_rates[24] - rev_rates[24])
              - (fwd_rates[25] - rev_rates[25]) - (fwd_rates[26] - rev_rates[26]) - (fwd_rates[27] - rev_rates[27])
              - (fwd_rates[28] - rev_rates[28]) - (fwd_rates[29] - rev_rates[29]) + (fwd_rates[30] - rev_rates[30])
              + (fwd_rates[37] - rev_rates[37]) + (fwd_rates[43] - rev_rates[43]) + (fwd_rates[85] - rev_rates[85])
              + (fwd_rates[121] - rev_rates[121]) + (fwd_rates[124] - rev_rates[124]) + (fwd_rates[154] - rev_rates[153])
              + (fwd_rates[177] - rev_rates[176]) + (fwd_rates[178] - rev_rates[177]) - (fwd_rates[180] - rev_rates[179])
              - (fwd_rates[181] - rev_rates[180]) + (fwd_rates[184] - rev_rates[183]) * pres_mod[27]
              - (fwd_rates[186] - rev_rates[185]) * pres_mod[28] - (fwd_rates[187] - rev_rates[186])
              - (fwd_rates[189] - rev_rates[188]) + (fwd_rates[193] - rev_rates[192]) - (fwd_rates[199] - rev_rates[198])
              - (fwd_rates[200] - rev_rates[199]) - (fwd_rates[206] - rev_rates[205]) - (fwd_rates[207] - rev_rates[206])
              - (fwd_rates[212] - rev_rates[211]) - (fwd_rates[216] - rev_rates[215]) + (fwd_rates[219] - rev_rates[218])
              - (fwd_rates[221] - rev_rates[220]) - (fwd_rates[230] - rev_rates[229]) - (fwd_rates[231] - rev_rates[230])
              - (fwd_rates[232] - rev_rates[231]) + (fwd_rates[243] - rev_rates[242]) + (fwd_rates[245] - rev_rates[244])
              - (fwd_rates[256] - rev_rates[255]) - (fwd_rates[257] - rev_rates[256]) + (fwd_rates[258] - rev_rates[257])
              - (fwd_rates[261] - rev_rates[260]) - (fwd_rates[262] - rev_rates[261]) - (fwd_rates[263] - rev_rates[262])
              - (fwd_rates[278] - rev_rates[277]) - fwd_rates[283] - (fwd_rates[284] - rev_rates[282])
              - (fwd_rates[285] - rev_rates[283]) + (fwd_rates[290] - rev_rates[286]) + (fwd_rates[293] - rev_rates[287])
              - (fwd_rates[295] - rev_rates[289]) - fwd_rates[296] - fwd_rates[304] - (fwd_rates[312] - rev_rates[297])
              - (fwd_rates[318] - rev_rates[303]);

  sp_rates[3] = (fwd_rates[0] - rev_rates[0]) * pres_mod[0] + (fwd_rates[3] - rev_rates[3])
              - (fwd_rates[30] - rev_rates[30]) - (fwd_rates[31] - rev_rates[31]) - (fwd_rates[32] - rev_rates[32]) * pres_mod[3]
              - (fwd_rates[33] - rev_rates[33]) - (fwd_rates[34] - rev_rates[34]) - (fwd_rates[35] - rev_rates[35])
              - (fwd_rates[36] - rev_rates[36]) - (fwd_rates[37] - rev_rates[37]) + (fwd_rates[44] - rev_rates[44])
              + (fwd_rates[86] - rev_rates[86]) + (fwd_rates[114] - rev_rates[114]) + (fwd_rates[115] - rev_rates[115])
              + (fwd_rates[117] - rev_rates[117]) - (fwd_rates[121] - rev_rates[121]) - (fwd_rates[124] - rev_rates[124])
              - fwd_rates[134] - (fwd_rates[143] - rev_rates[142]) - (fwd_rates[144] - rev_rates[143])
              - (fwd_rates[154] - rev_rates[153]) - (fwd_rates[155] - rev_rates[154]) - (fwd_rates[167] - rev_rates[166])
              - (fwd_rates[168] - rev_rates[167]) - (fwd_rates[169] - rev_rates[168]) - (fwd_rates[170] - rev_rates[169])
              - (fwd_rates[172] - rev_rates[171]) - (fwd_rates[174] - rev_rates[173]) - (fwd_rates[175] - rev_rates[174])
              - (fwd_rates[178] - rev_rates[177]) + (fwd_rates[180] - rev_rates[179]) + (fwd_rates[187] - rev_rates[186])
              - (fwd_rates[193] - rev_rates[192]) - (fwd_rates[194] - rev_rates[193]) - (fwd_rates[205] - rev_rates[204])
              - (fwd_rates[215] - rev_rates[214]) - (fwd_rates[219] - rev_rates[218]) - (fwd_rates[225] - rev_rates[224])
              - (fwd_rates[258] - rev_rates[257]) + (fwd_rates[286] - rev_rates[284]) - fwd_rates[289]
              - (fwd_rates[290] - rev_rates[286]) - (fwd_rates[293] - rev_rates[287]) - (fwd_rates[294] - rev_rates[288])
              - fwd_rates[297] - fwd_rates[305] - fwd_rates[306] + (fwd_rates[322] - rev_rates[307])
             ;

  sp_rates[4] = (fwd_rates[1] - rev_rates[1]) * pres_mod[1] + (fwd_rates[2] - rev_rates[2])
              + (fwd_rates[3] - rev_rates[3]) + (fwd_rates[4] - rev_rates[4]) + (fwd_rates[10] - rev_rates[10])
              + (fwd_rates[12] - rev_rates[12]) + (fwd_rates[14] - rev_rates[14]) + (fwd_rates[15] - rev_rates[15])
              + (fwd_rates[16] - rev_rates[16]) + (fwd_rates[17] - rev_rates[17]) + (fwd_rates[18] - rev_rates[18])
              + (fwd_rates[21] - rev_rates[21]) + (fwd_rates[26] - rev_rates[26]) + (fwd_rates[28] - rev_rates[28])
              + (fwd_rates[37] - rev_rates[37]) - (fwd_rates[42] - rev_rates[42]) * pres_mod[5]
              + 2.0 * (fwd_rates[45] - rev_rates[45]) + (fwd_rates[47] - rev_rates[47])
              + (fwd_rates[60] - rev_rates[60]) + (fwd_rates[65] - rev_rates[65]) - (fwd_rates[83] - rev_rates[83])
              - 2.0 * (fwd_rates[84] - rev_rates[84]) * pres_mod[19] - 2.0 * (fwd_rates[85] - rev_rates[85])
              - (fwd_rates[86] - rev_rates[86]) - (fwd_rates[87] - rev_rates[87]) - (fwd_rates[88] - rev_rates[88])
              - (fwd_rates[89] - rev_rates[89]) - (fwd_rates[90] - rev_rates[90]) - (fwd_rates[91] - rev_rates[91])
              - (fwd_rates[92] - rev_rates[92]) - (fwd_rates[93] - rev_rates[93]) - (fwd_rates[94] - rev_rates[94]) * pres_mod[20]
              - (fwd_rates[95] - rev_rates[95]) - (fwd_rates[96] - rev_rates[96]) - (fwd_rates[97] - rev_rates[97])
              - (fwd_rates[98] - rev_rates[98]) - (fwd_rates[99] - rev_rates[99]) - (fwd_rates[100] - rev_rates[100])
              - (fwd_rates[101] - rev_rates[101]) - (fwd_rates[102] - rev_rates[102]) - (fwd_rates[103] - rev_rates[103])
              - (fwd_rates[104] - rev_rates[104]) - (fwd_rates[105] - rev_rates[105]) - (fwd_rates[106] - rev_rates[106])
              - (fwd_rates[107] - rev_rates[107]) - (fwd_rates[108] - rev_rates[108]) - (fwd_rates[109] - rev_rates[109])
              - (fwd_rates[110] - rev_rates[110]) - (fwd_rates[111] - rev_rates[111]) - (fwd_rates[112] - rev_rates[112])
              - (fwd_rates[113] - rev_rates[113]) + (fwd_rates[116] - rev_rates[116]) + (fwd_rates[118] - rev_rates[118])
              + (fwd_rates[119] - rev_rates[119]) + fwd_rates[134] + (fwd_rates[143] - rev_rates[142])
              + (fwd_rates[155] - rev_rates[154]) + (fwd_rates[175] - rev_rates[174]) - (fwd_rates[179] - rev_rates[178])
              + (fwd_rates[182] - rev_rates[181]) - (fwd_rates[183] - rev_rates[182]) + (fwd_rates[185] - rev_rates[184])
              + (fwd_rates[188] - rev_rates[187]) - (fwd_rates[191] - rev_rates[190]) - (fwd_rates[192] - rev_rates[191])
              + (fwd_rates[194] - rev_rates[193]) + (fwd_rates[197] - rev_rates[196]) + (fwd_rates[199] - rev_rates[198])
              - (fwd_rates[202] - rev_rates[201]) + (fwd_rates[206] - rev_rates[205]) - (fwd_rates[209] - rev_rates[208])
              + (fwd_rates[212] - rev_rates[211]) - (fwd_rates[214] - rev_rates[213]) - (fwd_rates[217] - rev_rates[216])
              + (fwd_rates[218] - rev_rates[217]) - (fwd_rates[223] - rev_rates[222]) + (fwd_rates[232] - rev_rates[231])
              - (fwd_rates[233] - rev_rates[232]) - (fwd_rates[234] - rev_rates[233]) - (fwd_rates[235] - rev_rates[234])
              + (fwd_rates[249] - rev_rates[248]) + (fwd_rates[252] - rev_rates[251]) + (fwd_rates[255] - rev_rates[254])
              - (fwd_rates[259] - rev_rates[258]) + (fwd_rates[263] - rev_rates[262]) - (fwd_rates[266] - rev_rates[265])
              - (fwd_rates[267] - rev_rates[266]) + (fwd_rates[270] - rev_rates[269]) - (fwd_rates[277] - rev_rates[276])
              + (fwd_rates[278] - rev_rates[277]) - (fwd_rates[286] - rev_rates[284]) - fwd_rates[287]
              + (fwd_rates[295] - rev_rates[289]) + fwd_rates[296] - fwd_rates[300] + fwd_rates[305]
              + fwd_rates[306] - (fwd_rates[309] - rev_rates[294]) - (fwd_rates[310] - rev_rates[295])
              + (fwd_rates[312] - rev_rates[297]) - (fwd_rates[314] - rev_rates[299]) - (fwd_rates[321] - rev_rates[306])
              + fwd_rates[323];

  sp_rates[5] = (fwd_rates[42] - rev_rates[42]) * pres_mod[5] + (fwd_rates[43] - rev_rates[43])
              + (fwd_rates[47] - rev_rates[47]) + (fwd_rates[61] - rev_rates[61]) + (fwd_rates[66] - rev_rates[66])
              + (fwd_rates[83] - rev_rates[83]) + (fwd_rates[85] - rev_rates[85]) + (fwd_rates[86] - rev_rates[86])
              + (fwd_rates[87] - rev_rates[87]) + (fwd_rates[88] - rev_rates[88]) + (fwd_rates[92] - rev_rates[92])
              + (fwd_rates[95] - rev_rates[95]) + (fwd_rates[96] - rev_rates[96]) + (fwd_rates[97] - rev_rates[97])
              + (fwd_rates[99] - rev_rates[99]) + (fwd_rates[100] - rev_rates[100]) + (fwd_rates[101] - rev_rates[101])
              + (fwd_rates[102] - rev_rates[102]) + (fwd_rates[103] - rev_rates[103]) + (fwd_rates[104] - rev_rates[104])
              + (fwd_rates[108] - rev_rates[108]) + (fwd_rates[110] - rev_rates[110]) + (fwd_rates[111] - rev_rates[111])
              + (fwd_rates[112] - rev_rates[112]) + (fwd_rates[113] - rev_rates[113]) - (fwd_rates[126] - rev_rates[126])
              + (fwd_rates[144] - rev_rates[143]) - (fwd_rates[146] - rev_rates[145]) * pres_mod[23]
              + (fwd_rates[192] - rev_rates[191]) - (fwd_rates[196] - rev_rates[195]) + (fwd_rates[202] - rev_rates[201])
              + (fwd_rates[209] - rev_rates[208]) + (fwd_rates[214] - rev_rates[213]) - (fwd_rates[218] - rev_rates[217])
              + (fwd_rates[254] - rev_rates[253]) + (fwd_rates[266] - rev_rates[265]) + (fwd_rates[277] - rev_rates[276])
              + (fwd_rates[286] - rev_rates[284]) - fwd_rates[292] + fwd_rates[300] + (fwd_rates[309] - rev_rates[294])
              + (fwd_rates[314] - rev_rates[299]);

  sp_rates[6] = -(fwd_rates[3] - rev_rates[3]) + (fwd_rates[4] - rev_rates[4]) + (fwd_rates[31] - rev_rates[31])
              + (fwd_rates[32] - rev_rates[32]) * pres_mod[3] + (fwd_rates[33] - rev_rates[33])
              + (fwd_rates[34] - rev_rates[34]) + (fwd_rates[35] - rev_rates[35]) + (fwd_rates[36] - rev_rates[36])
              - (fwd_rates[43] - rev_rates[43]) - (fwd_rates[44] - rev_rates[44]) - (fwd_rates[45] - rev_rates[45])
              + (fwd_rates[46] - rev_rates[46]) - (fwd_rates[86] - rev_rates[86]) + (fwd_rates[87] - rev_rates[87])
              + (fwd_rates[88] - rev_rates[88]) - 2.0 * (fwd_rates[114] - rev_rates[114])
              - 2.0 * (fwd_rates[115] - rev_rates[115]) - (fwd_rates[116] - rev_rates[116])
              - (fwd_rates[117] - rev_rates[117]) - (fwd_rates[118] - rev_rates[118]) - (fwd_rates[119] - rev_rates[119])
              - (fwd_rates[120] - rev_rates[120]) + (fwd_rates[156] - rev_rates[155]) + (fwd_rates[167] - rev_rates[166])
              + (fwd_rates[168] - rev_rates[167]) + (fwd_rates[169] - rev_rates[168]) + (fwd_rates[174] - rev_rates[173])
              + (fwd_rates[183] - rev_rates[182]) - (fwd_rates[185] - rev_rates[184]) + (fwd_rates[205] - rev_rates[204])
              + (fwd_rates[215] - rev_rates[214]) - (fwd_rates[286] - rev_rates[284]) + (fwd_rates[294] - rev_rates[288])
              + fwd_rates[297] - fwd_rates[301] + (fwd_rates[315] - rev_rates[300]) - (fwd_rates[322] - rev_rates[307])
              - fwd_rates[323];

  sp_rates[7] = -(fwd_rates[4] - rev_rates[4]) - (fwd_rates[46] - rev_rates[46]) - (fwd_rates[47] - rev_rates[47])
              + (fwd_rates[84] - rev_rates[84]) * pres_mod[19] - (fwd_rates[87] - rev_rates[87])
              - (fwd_rates[88] - rev_rates[88]) + (fwd_rates[114] - rev_rates[114]) + (fwd_rates[115] - rev_rates[115])
              + (fwd_rates[120] - rev_rates[120]) - (fwd_rates[156] - rev_rates[155]) + fwd_rates[301]
              - (fwd_rates[315] - rev_rates[300]);

  sp_rates[8] = (fwd_rates[48] - rev_rates[48]) - (fwd_rates[89] - rev_rates[89]) - (fwd_rates[121] - rev_rates[121])
              - (fwd_rates[122] - rev_rates[122]) - (fwd_rates[123] - rev_rates[123]) - (fwd_rates[238] - rev_rates[237])
              - (fwd_rates[243] - rev_rates[242]) - (fwd_rates[244] - rev_rates[243]);

  sp_rates[9] = -(fwd_rates[5] - rev_rates[5]) + (fwd_rates[19] - rev_rates[19]) - (fwd_rates[48] - rev_rates[48])
              + (fwd_rates[50] - rev_rates[50]) - (fwd_rates[90] - rev_rates[90]) + (fwd_rates[92] - rev_rates[92])
              - (fwd_rates[124] - rev_rates[124]) - (fwd_rates[125] - rev_rates[125]) - (fwd_rates[126] - rev_rates[126])
              - (fwd_rates[127] - rev_rates[127]) - (fwd_rates[128] - rev_rates[128]) - (fwd_rates[129] - rev_rates[129])
              - (fwd_rates[130] - rev_rates[130]) * pres_mod[21] - (fwd_rates[131] - rev_rates[131])
              - (fwd_rates[132] - rev_rates[132]) - (fwd_rates[133] - rev_rates[133]) - (fwd_rates[239] - rev_rates[238])
              - (fwd_rates[240] - rev_rates[239]) * pres_mod[34] - (fwd_rates[245] - rev_rates[244])
              - (fwd_rates[246] - rev_rates[245]) - (fwd_rates[247] - rev_rates[246]) - (fwd_rates[288] - rev_rates[285]) * pres_mod[36]
             ;

  sp_rates[10] = -(fwd_rates[6] - rev_rates[6]) + (fwd_rates[22] - rev_rates[22]) + (fwd_rates[29] - rev_rates[29])
               - (fwd_rates[49] - rev_rates[49]) * pres_mod[6] - (fwd_rates[91] - rev_rates[91])
               - (fwd_rates[92] - rev_rates[92]) + (fwd_rates[95] - rev_rates[95]) - (fwd_rates[116] - rev_rates[116])
               - (fwd_rates[122] - rev_rates[122]) + (fwd_rates[125] - rev_rates[125])
               - (fwd_rates[127] - rev_rates[127]) - fwd_rates[134] - (fwd_rates[135] - rev_rates[134])
               - 2.0 * (fwd_rates[136] - rev_rates[135]) - (fwd_rates[137] - rev_rates[136])
               - (fwd_rates[138] - rev_rates[137]) - (fwd_rates[139] - rev_rates[138]) * pres_mod[22]
               - (fwd_rates[140] - rev_rates[139]) + (fwd_rates[141] - rev_rates[140])
               + (fwd_rates[142] - rev_rates[141]) + (fwd_rates[147] - rev_rates[146])
               + (fwd_rates[150] - rev_rates[149]) + (fwd_rates[151] - rev_rates[150])
               + (fwd_rates[237] - rev_rates[236]) - (fwd_rates[241] - rev_rates[240])
               - (fwd_rates[248] - rev_rates[247]) - (fwd_rates[249] - rev_rates[248])
               - (fwd_rates[250] - rev_rates[249]) + (fwd_rates[260] - rev_rates[259])
               - fwd_rates[289] - (fwd_rates[290] - rev_rates[286]) - 2.0 * fwd_rates[291]
               + fwd_rates[304];

  sp_rates[11] = -(fwd_rates[7] - rev_rates[7]) - (fwd_rates[8] - rev_rates[8]) - (fwd_rates[50] - rev_rates[50])
               + (fwd_rates[61] - rev_rates[61]) + (fwd_rates[66] - rev_rates[66]) + (fwd_rates[78] - rev_rates[78])
               - (fwd_rates[93] - rev_rates[93]) + (fwd_rates[96] - rev_rates[96]) - (fwd_rates[141] - rev_rates[140])
               - (fwd_rates[142] - rev_rates[141]) - (fwd_rates[143] - rev_rates[142])
               - (fwd_rates[144] - rev_rates[143]) - (fwd_rates[145] - rev_rates[144])
               - (fwd_rates[146] - rev_rates[145]) * pres_mod[23] - (fwd_rates[147] - rev_rates[146])
               - (fwd_rates[148] - rev_rates[147]) - (fwd_rates[149] - rev_rates[148])
               - (fwd_rates[150] - rev_rates[149]) - (fwd_rates[151] - rev_rates[150])
               - (fwd_rates[152] - rev_rates[151]) - (fwd_rates[153] - rev_rates[152])
               - (fwd_rates[242] - rev_rates[241]) - (fwd_rates[251] - rev_rates[250])
               - (fwd_rates[252] - rev_rates[251]) - (fwd_rates[253] - rev_rates[252])
               - fwd_rates[292];

  sp_rates[12] = -(fwd_rates[9] - rev_rates[9]) + (fwd_rates[10] - rev_rates[10]) + (fwd_rates[24] - rev_rates[24])
               + (fwd_rates[25] - rev_rates[25]) + (fwd_rates[49] - rev_rates[49]) * pres_mod[6]
               - (fwd_rates[51] - rev_rates[51]) * pres_mod[7] + (fwd_rates[52] - rev_rates[52])
               + (fwd_rates[60] - rev_rates[60]) + (fwd_rates[65] - rev_rates[65]) + (fwd_rates[80] - rev_rates[80])
               - (fwd_rates[94] - rev_rates[94]) * pres_mod[20] - (fwd_rates[95] - rev_rates[95])
               - (fwd_rates[96] - rev_rates[96]) + (fwd_rates[97] - rev_rates[97]) + (fwd_rates[109] - rev_rates[109])
               - (fwd_rates[117] - rev_rates[117]) - (fwd_rates[118] - rev_rates[118])
               - (fwd_rates[123] - rev_rates[123]) - (fwd_rates[128] - rev_rates[128])
               + (fwd_rates[135] - rev_rates[134]) - (fwd_rates[137] - rev_rates[136])
               + 2.0 * (fwd_rates[138] - rev_rates[137]) + (fwd_rates[145] - rev_rates[144])
               - (fwd_rates[148] - rev_rates[147]) + 2.0 * (fwd_rates[149] - rev_rates[148])
               + (fwd_rates[153] - rev_rates[152]) - (fwd_rates[154] - rev_rates[153])
               - (fwd_rates[155] - rev_rates[154]) - (fwd_rates[156] - rev_rates[155])
               - 2.0 * (fwd_rates[157] - rev_rates[156]) * pres_mod[24] - 2.0 * (fwd_rates[158] - rev_rates[157])
               - (fwd_rates[159] - rev_rates[158]) - (fwd_rates[160] - rev_rates[159])
               - (fwd_rates[161] - rev_rates[160]) - (fwd_rates[162] - rev_rates[161])
               - (fwd_rates[163] - rev_rates[162]) - (fwd_rates[164] - rev_rates[163])
               - (fwd_rates[210] - rev_rates[209]) - (fwd_rates[254] - rev_rates[253])
               - (fwd_rates[255] - rev_rates[254]) - (fwd_rates[274] - rev_rates[273])
               - (fwd_rates[275] - rev_rates[274]) - fwd_rates[283] - fwd_rates[287] + (fwd_rates[288] - rev_rates[285]) * pres_mod[36]
               + fwd_rates[296] + fwd_rates[297] + fwd_rates[299] + fwd_rates[300] + fwd_rates[301]
               + (fwd_rates[307] - rev_rates[292]) - (fwd_rates[311] - rev_rates[296]) * pres_mod[38]
               - (fwd_rates[316] - rev_rates[301]) - (fwd_rates[317] - rev_rates[302]) * pres_mod[39]
               + (fwd_rates[320] - rev_rates[305]) - (fwd_rates[324] - rev_rates[308]);

  sp_rates[13] = -(fwd_rates[10] - rev_rates[10]) + (fwd_rates[51] - rev_rates[51]) * pres_mod[7]
               - (fwd_rates[52] - rev_rates[52]) - (fwd_rates[97] - rev_rates[97]) + (fwd_rates[117] - rev_rates[117])
               - (fwd_rates[129] - rev_rates[129]) - (fwd_rates[138] - rev_rates[137])
               - (fwd_rates[149] - rev_rates[148]) + (fwd_rates[156] - rev_rates[155])
               + (fwd_rates[159] - rev_rates[158]) + (fwd_rates[160] - rev_rates[159])
               + (fwd_rates[161] - rev_rates[160]) + (fwd_rates[162] - rev_rates[161])
               + (fwd_rates[163] - rev_rates[162]) + (fwd_rates[164] - rev_rates[163])
               + (fwd_rates[210] - rev_rates[209]) + fwd_rates[302] + (fwd_rates[316] - rev_rates[301])
              ;

  sp_rates[14] = (fwd_rates[5] - rev_rates[5]) + (fwd_rates[7] - rev_rates[7]) - (fwd_rates[11] - rev_rates[11]) * pres_mod[2]
               + (fwd_rates[12] - rev_rates[12]) + (fwd_rates[19] - rev_rates[19]) + (fwd_rates[22] - rev_rates[22])
               + 2.0 * (fwd_rates[27] - rev_rates[27]) - (fwd_rates[30] - rev_rates[30])
               + (fwd_rates[54] - rev_rates[54]) + (fwd_rates[78] - rev_rates[78]) + (fwd_rates[80] - rev_rates[80])
               - (fwd_rates[82] - rev_rates[82]) * pres_mod[18] + (fwd_rates[89] - rev_rates[89])
               - (fwd_rates[98] - rev_rates[98]) + (fwd_rates[99] - rev_rates[99]) + (fwd_rates[109] - rev_rates[109])
               - (fwd_rates[119] - rev_rates[119]) + (fwd_rates[121] - rev_rates[121])
               - (fwd_rates[130] - rev_rates[130]) * pres_mod[21] + (fwd_rates[131] - rev_rates[131])
               + (fwd_rates[133] - rev_rates[133]) + fwd_rates[134] - (fwd_rates[139] - rev_rates[138]) * pres_mod[22]
               + (fwd_rates[140] - rev_rates[139]) + (fwd_rates[143] - rev_rates[142])
               + (fwd_rates[144] - rev_rates[143]) + (fwd_rates[152] - rev_rates[151])
               + (fwd_rates[159] - rev_rates[158]) + (fwd_rates[165] - rev_rates[164])
               + (fwd_rates[166] - rev_rates[165]) * pres_mod[25] + (fwd_rates[167] - rev_rates[166])
               + (fwd_rates[170] - rev_rates[169]) + 2.0 * (fwd_rates[175] - rev_rates[174])
               + 2.0 * (fwd_rates[176] - rev_rates[175]) + (fwd_rates[216] - rev_rates[215])
               + (fwd_rates[221] - rev_rates[220]) + (fwd_rates[222] - rev_rates[221])
               + (fwd_rates[223] - rev_rates[222]) + (fwd_rates[224] - rev_rates[223])
               + (fwd_rates[226] - rev_rates[225]) * pres_mod[31] + (fwd_rates[227] - rev_rates[226])
               + (fwd_rates[231] - rev_rates[230]) + (fwd_rates[235] - rev_rates[234])
               + (fwd_rates[244] - rev_rates[243]) + (fwd_rates[256] - rev_rates[255])
               + (fwd_rates[262] - rev_rates[261]) + (fwd_rates[264] - rev_rates[263])
               + (fwd_rates[268] - rev_rates[267]) * pres_mod[35] + (fwd_rates[271] - rev_rates[270])
               + (fwd_rates[273] - rev_rates[272]) + (fwd_rates[279] - rev_rates[278])
               + (fwd_rates[282] - rev_rates[281]) + fwd_rates[283] + fwd_rates[296] + fwd_rates[297]
               + fwd_rates[299] + fwd_rates[300] + fwd_rates[301] + fwd_rates[302] + fwd_rates[305]
              ;

  sp_rates[15] = (fwd_rates[11] - rev_rates[11]) * pres_mod[2] + (fwd_rates[13] - rev_rates[13])
               + (fwd_rates[29] - rev_rates[29]) + (fwd_rates[30] - rev_rates[30]) + (fwd_rates[98] - rev_rates[98])
               + (fwd_rates[119] - rev_rates[119]) - (fwd_rates[131] - rev_rates[131])
               - (fwd_rates[152] - rev_rates[151]) + (fwd_rates[225] - rev_rates[224])
               + (fwd_rates[228] - rev_rates[227]) + (fwd_rates[261] - rev_rates[260])
               + (fwd_rates[267] - rev_rates[266]) - (fwd_rates[279] - rev_rates[278])
               + (fwd_rates[281] - rev_rates[280]) - (fwd_rates[282] - rev_rates[281])
               + fwd_rates[289] + fwd_rates[304];

  sp_rates[16] = (fwd_rates[6] - rev_rates[6]) + (fwd_rates[8] - rev_rates[8]) - (fwd_rates[12] - rev_rates[12])
               - (fwd_rates[13] - rev_rates[13]) + (fwd_rates[14] - rev_rates[14]) + (fwd_rates[24] - rev_rates[24])
               + (fwd_rates[31] - rev_rates[31]) - (fwd_rates[53] - rev_rates[53]) * pres_mod[8]
               - (fwd_rates[54] - rev_rates[54]) + (fwd_rates[57] - rev_rates[57]) + (fwd_rates[90] - rev_rates[90])
               - (fwd_rates[99] - rev_rates[99]) + (fwd_rates[100] - rev_rates[100]) + (fwd_rates[120] - rev_rates[120])
               + (fwd_rates[124] - rev_rates[124]) + (fwd_rates[131] - rev_rates[131])
               - (fwd_rates[159] - rev_rates[158]) + (fwd_rates[160] - rev_rates[159])
               - (fwd_rates[165] - rev_rates[164]) - (fwd_rates[166] - rev_rates[165]) * pres_mod[25]
               - (fwd_rates[167] - rev_rates[166]) + (fwd_rates[170] - rev_rates[169])
               + (fwd_rates[172] - rev_rates[171]) + (fwd_rates[247] - rev_rates[246])
               + (fwd_rates[258] - rev_rates[257]) + (fwd_rates[259] - rev_rates[258])
               + 2.0 * fwd_rates[306] + (fwd_rates[307] - rev_rates[292]) + (fwd_rates[310] - rev_rates[295])
              ;

  sp_rates[17] = (fwd_rates[9] - rev_rates[9]) - (fwd_rates[14] - rev_rates[14]) + (fwd_rates[15] - rev_rates[15])
               + (fwd_rates[16] - rev_rates[16]) + (fwd_rates[25] - rev_rates[25]) - (fwd_rates[31] - rev_rates[31])
               + (fwd_rates[53] - rev_rates[53]) * pres_mod[8] - (fwd_rates[55] - rev_rates[55]) * pres_mod[9]
               - (fwd_rates[56] - rev_rates[56]) * pres_mod[10] - (fwd_rates[57] - rev_rates[57])
               + (fwd_rates[59] - rev_rates[59]) + (fwd_rates[64] - rev_rates[64]) + (fwd_rates[82] - rev_rates[82]) * pres_mod[18]
               + (fwd_rates[91] - rev_rates[91]) + (fwd_rates[93] - rev_rates[93]) - (fwd_rates[100] - rev_rates[100])
               + (fwd_rates[101] - rev_rates[101]) + (fwd_rates[102] - rev_rates[102])
               + (fwd_rates[116] - rev_rates[116]) - (fwd_rates[120] - rev_rates[120])
               + (fwd_rates[126] - rev_rates[126]) - (fwd_rates[132] - rev_rates[132])
               + (fwd_rates[152] - rev_rates[151]) + (fwd_rates[155] - rev_rates[154])
               - (fwd_rates[160] - rev_rates[159]) + (fwd_rates[168] - rev_rates[167])
               + (fwd_rates[169] - rev_rates[168]) + (fwd_rates[172] - rev_rates[171])
               + fwd_rates[287] + (fwd_rates[290] - rev_rates[286]) + fwd_rates[292] + fwd_rates[305]
               + (fwd_rates[318] - rev_rates[303]) + fwd_rates[323];

  sp_rates[18] = -(fwd_rates[15] - rev_rates[15]) + (fwd_rates[17] - rev_rates[17]) + (fwd_rates[55] - rev_rates[55]) * pres_mod[9]
               - (fwd_rates[58] - rev_rates[58]) * pres_mod[11] - (fwd_rates[59] - rev_rates[59])
               - (fwd_rates[60] - rev_rates[60]) - (fwd_rates[61] - rev_rates[61]) + (fwd_rates[63] - rev_rates[63])
               + (fwd_rates[67] - rev_rates[67]) - (fwd_rates[101] - rev_rates[101]) + (fwd_rates[103] - rev_rates[103])
               + (fwd_rates[161] - rev_rates[160]) - (fwd_rates[168] - rev_rates[167])
               + (fwd_rates[310] - rev_rates[295]) + (fwd_rates[321] - rev_rates[306])
              ;

  sp_rates[19] = -(fwd_rates[16] - rev_rates[16]) + (fwd_rates[18] - rev_rates[18]) + (fwd_rates[56] - rev_rates[56]) * pres_mod[10]
               - (fwd_rates[62] - rev_rates[62]) * pres_mod[12] - (fwd_rates[63] - rev_rates[63])
               - (fwd_rates[64] - rev_rates[64]) - (fwd_rates[65] - rev_rates[65]) - (fwd_rates[66] - rev_rates[66])
               + (fwd_rates[68] - rev_rates[68]) - (fwd_rates[102] - rev_rates[102]) + (fwd_rates[104] - rev_rates[104])
               + (fwd_rates[118] - rev_rates[118]) + (fwd_rates[154] - rev_rates[153])
               + (fwd_rates[162] - rev_rates[161]) - (fwd_rates[169] - rev_rates[168])
              ;

  sp_rates[20] = -(fwd_rates[17] - rev_rates[17]) - (fwd_rates[18] - rev_rates[18]) + (fwd_rates[58] - rev_rates[58]) * pres_mod[11]
               + (fwd_rates[62] - rev_rates[62]) * pres_mod[12] - (fwd_rates[67] - rev_rates[67])
               - (fwd_rates[68] - rev_rates[68]) + (fwd_rates[94] - rev_rates[94]) * pres_mod[20]
               - (fwd_rates[103] - rev_rates[103]) - (fwd_rates[104] - rev_rates[104])
               + (fwd_rates[146] - rev_rates[145]) * pres_mod[23] - (fwd_rates[161] - rev_rates[160])
               - (fwd_rates[162] - rev_rates[161]);

  sp_rates[21] = -(fwd_rates[19] - rev_rates[19]) + (fwd_rates[21] - rev_rates[21]) - (fwd_rates[69] - rev_rates[69]) * pres_mod[13]
               - (fwd_rates[105] - rev_rates[105]) + (fwd_rates[108] - rev_rates[108])
               + (fwd_rates[122] - rev_rates[122]) - (fwd_rates[170] - rev_rates[169])
               - (fwd_rates[171] - rev_rates[170]);

  sp_rates[22] = -(fwd_rates[20] - rev_rates[20]) - (fwd_rates[21] - rev_rates[21]) - (fwd_rates[22] - rev_rates[22])
               + (fwd_rates[69] - rev_rates[69]) * pres_mod[13] - (fwd_rates[70] - rev_rates[70]) * pres_mod[14]
               + (fwd_rates[72] - rev_rates[72]) - (fwd_rates[106] - rev_rates[106]) - (fwd_rates[107] - rev_rates[107])
               - (fwd_rates[108] - rev_rates[108]) - (fwd_rates[109] - rev_rates[109])
               + (fwd_rates[110] - rev_rates[110]) + (fwd_rates[123] - rev_rates[123])
               + (fwd_rates[127] - rev_rates[127]) + (fwd_rates[133] - rev_rates[133])
               + (fwd_rates[136] - rev_rates[135]) + (fwd_rates[171] - rev_rates[170])
               + (fwd_rates[173] - rev_rates[172]) * pres_mod[26] + (fwd_rates[176] - rev_rates[175])
               + fwd_rates[291] + (fwd_rates[294] - rev_rates[288]);

  sp_rates[23] = -(fwd_rates[23] - rev_rates[23]) + (fwd_rates[70] - rev_rates[70]) * pres_mod[14]
               - (fwd_rates[71] - rev_rates[71]) * pres_mod[15] - (fwd_rates[72] - rev_rates[72])
               + (fwd_rates[74] - rev_rates[74]) - (fwd_rates[110] - rev_rates[110]) + (fwd_rates[111] - rev_rates[111])
               + (fwd_rates[128] - rev_rates[128]) + (fwd_rates[140] - rev_rates[139])
               + (fwd_rates[163] - rev_rates[162]) - (fwd_rates[172] - rev_rates[171])
               - (fwd_rates[293] - rev_rates[287]) - (fwd_rates[294] - rev_rates[288])
              ;

  sp_rates[24] = -(fwd_rates[24] - rev_rates[24]) + (fwd_rates[71] - rev_rates[71]) * pres_mod[15]
               - (fwd_rates[73] - rev_rates[73]) * pres_mod[16] - (fwd_rates[74] - rev_rates[74])
               + (fwd_rates[76] - rev_rates[76]) - (fwd_rates[111] - rev_rates[111]) + (fwd_rates[129] - rev_rates[129])
               + (fwd_rates[137] - rev_rates[136]) + (fwd_rates[148] - rev_rates[147])
               - (fwd_rates[163] - rev_rates[162]) - (fwd_rates[173] - rev_rates[172]) * pres_mod[26]
               + (fwd_rates[174] - rev_rates[173]) - (fwd_rates[284] - rev_rates[282])
               - (fwd_rates[317] - rev_rates[302]) * pres_mod[39];

  sp_rates[25] = -(fwd_rates[25] - rev_rates[25]) + (fwd_rates[26] - rev_rates[26]) + (fwd_rates[73] - rev_rates[73]) * pres_mod[16]
               - (fwd_rates[75] - rev_rates[75]) * pres_mod[17] - (fwd_rates[76] - rev_rates[76])
               + (fwd_rates[77] - rev_rates[77]) + (fwd_rates[112] - rev_rates[112]) + (fwd_rates[153] - rev_rates[152])
               + (fwd_rates[158] - rev_rates[157]) + (fwd_rates[164] - rev_rates[163])
               - (fwd_rates[174] - rev_rates[173]) - (fwd_rates[285] - rev_rates[283])
               - (fwd_rates[311] - rev_rates[296]) * pres_mod[38] + (fwd_rates[318] - rev_rates[303])
               + (fwd_rates[320] - rev_rates[305]) + (fwd_rates[321] - rev_rates[306])
               + fwd_rates[323] + 2.0 * (fwd_rates[324] - rev_rates[308]);

  sp_rates[26] = -(fwd_rates[26] - rev_rates[26]) + (fwd_rates[75] - rev_rates[75]) * pres_mod[17]
               - (fwd_rates[77] - rev_rates[77]) - (fwd_rates[112] - rev_rates[112]) - (fwd_rates[153] - rev_rates[152])
               + (fwd_rates[157] - rev_rates[156]) * pres_mod[24] - (fwd_rates[164] - rev_rates[163])
              ;

  sp_rates[27] = (fwd_rates[20] - rev_rates[20]) - (fwd_rates[27] - rev_rates[27]) + (fwd_rates[28] - rev_rates[28])
               - (fwd_rates[78] - rev_rates[78]) + (fwd_rates[79] - rev_rates[79]) + (fwd_rates[105] - rev_rates[105])
               + (fwd_rates[113] - rev_rates[113]) + (fwd_rates[130] - rev_rates[130]) * pres_mod[21]
               - (fwd_rates[133] - rev_rates[133]) - (fwd_rates[140] - rev_rates[139])
               - (fwd_rates[175] - rev_rates[174]) - 2.0 * (fwd_rates[176] - rev_rates[175])
               - (fwd_rates[273] - rev_rates[272]);

  sp_rates[28] = (fwd_rates[23] - rev_rates[23]) - (fwd_rates[28] - rev_rates[28]) - (fwd_rates[29] - rev_rates[29])
               - (fwd_rates[79] - rev_rates[79]) - (fwd_rates[80] - rev_rates[80]) + (fwd_rates[81] - rev_rates[81])
               + (fwd_rates[106] - rev_rates[106]) - (fwd_rates[113] - rev_rates[113])
               + (fwd_rates[132] - rev_rates[132]) + (fwd_rates[139] - rev_rates[138]) * pres_mod[22]
               - (fwd_rates[303] - rev_rates[291]) * pres_mod[37] + (fwd_rates[308] - rev_rates[293])
               + (fwd_rates[309] - rev_rates[294]);

  sp_rates[29] = -(fwd_rates[81] - rev_rates[81]) + (fwd_rates[107] - rev_rates[107]);

  sp_rates[30] = -(fwd_rates[177] - rev_rates[176]) - (fwd_rates[178] - rev_rates[177])
               - (fwd_rates[179] - rev_rates[178]) + (fwd_rates[190] - rev_rates[189])
               + (fwd_rates[192] - rev_rates[191]) - (fwd_rates[195] - rev_rates[194])
               + (fwd_rates[216] - rev_rates[215]) - (fwd_rates[224] - rev_rates[223])
               + (fwd_rates[226] - rev_rates[225]) * pres_mod[31] - (fwd_rates[237] - rev_rates[236])
               + (fwd_rates[238] - rev_rates[237]) + (fwd_rates[239] - rev_rates[238])
               + (fwd_rates[244] - rev_rates[243]) + (fwd_rates[247] - rev_rates[246])
               - (fwd_rates[274] - rev_rates[273]) - (fwd_rates[275] - rev_rates[274])
               - (fwd_rates[282] - rev_rates[281]);

  sp_rates[31] = -(fwd_rates[189] - rev_rates[188]) - (fwd_rates[190] - rev_rates[189])
               - (fwd_rates[191] - rev_rates[190]) - (fwd_rates[192] - rev_rates[191])
               - (fwd_rates[193] - rev_rates[192]) - (fwd_rates[194] - rev_rates[193])
               - (fwd_rates[195] - rev_rates[194]) - (fwd_rates[196] - rev_rates[195])
               - (fwd_rates[197] - rev_rates[196]) - (fwd_rates[198] - rev_rates[197])
               + (fwd_rates[199] - rev_rates[198]) + (fwd_rates[201] - rev_rates[200])
               + (fwd_rates[202] - rev_rates[201]) + (fwd_rates[207] - rev_rates[206])
               + (fwd_rates[222] - rev_rates[221]) + (fwd_rates[231] - rev_rates[230])
               + (fwd_rates[241] - rev_rates[240]) + (fwd_rates[242] - rev_rates[241])
               + (fwd_rates[261] - rev_rates[260]) + (fwd_rates[268] - rev_rates[267]) * pres_mod[35]
               - (fwd_rates[279] - rev_rates[278]);

  sp_rates[32] = -(fwd_rates[199] - rev_rates[198]) - (fwd_rates[200] - rev_rates[199])
               - (fwd_rates[201] - rev_rates[200]) - (fwd_rates[202] - rev_rates[201])
               + (fwd_rates[235] - rev_rates[234]) + (fwd_rates[264] - rev_rates[263])
               + (fwd_rates[267] - rev_rates[266]) + (fwd_rates[271] - rev_rates[270])
               + (fwd_rates[276] - rev_rates[275]) + (fwd_rates[277] - rev_rates[276])
               + (fwd_rates[278] - rev_rates[277]);

  sp_rates[33] = -(fwd_rates[276] - rev_rates[275]) - (fwd_rates[277] - rev_rates[276])
               - (fwd_rates[278] - rev_rates[277]);

  sp_rates[34] = -(fwd_rates[203] - rev_rates[202]) - (fwd_rates[204] - rev_rates[203]) * pres_mod[29]
               - (fwd_rates[205] - rev_rates[204]) - (fwd_rates[206] - rev_rates[205])
               - (fwd_rates[207] - rev_rates[206]) - (fwd_rates[208] - rev_rates[207])
               - (fwd_rates[209] - rev_rates[208]) - (fwd_rates[210] - rev_rates[209])
              ;

  sp_rates[35] = -(fwd_rates[177] - rev_rates[176]) + (fwd_rates[178] - rev_rates[177])
               + (fwd_rates[179] - rev_rates[178]) + 2.0 * (fwd_rates[181] - rev_rates[180])
               - (fwd_rates[185] - rev_rates[184]) - (fwd_rates[186] - rev_rates[185]) * pres_mod[28]
               + (fwd_rates[187] - rev_rates[186]) + (fwd_rates[188] - rev_rates[187])
               + (fwd_rates[189] - rev_rates[188]) + (fwd_rates[194] - rev_rates[193])
               - (fwd_rates[197] - rev_rates[196]) - (fwd_rates[198] - rev_rates[197])
               + (fwd_rates[207] - rev_rates[206]) - (fwd_rates[211] - rev_rates[210]) * pres_mod[30]
               + (fwd_rates[212] - rev_rates[211]) + (fwd_rates[213] - rev_rates[212])
               + (fwd_rates[214] - rev_rates[213]) + (fwd_rates[215] - rev_rates[214])
               + (fwd_rates[221] - rev_rates[220]) + (fwd_rates[223] - rev_rates[222])
               + (fwd_rates[225] - rev_rates[224]) - (fwd_rates[227] - rev_rates[226])
               - (fwd_rates[228] - rev_rates[227]) - (fwd_rates[243] - rev_rates[242])
               - (fwd_rates[244] - rev_rates[243]) - (fwd_rates[245] - rev_rates[244])
               - (fwd_rates[246] - rev_rates[245]) - (fwd_rates[247] - rev_rates[246])
               - (fwd_rates[248] - rev_rates[247]) - (fwd_rates[249] - rev_rates[248])
               - (fwd_rates[250] - rev_rates[249]) - (fwd_rates[251] - rev_rates[250])
               - (fwd_rates[252] - rev_rates[251]) - (fwd_rates[253] - rev_rates[252])
               - (fwd_rates[254] - rev_rates[253]) - (fwd_rates[255] - rev_rates[254])
               + (fwd_rates[257] - rev_rates[256]) - (fwd_rates[273] - rev_rates[272])
               + (fwd_rates[280] - rev_rates[279]) + (fwd_rates[282] - rev_rates[281])
              ;

  sp_rates[36] = (fwd_rates[185] - rev_rates[184]) + (fwd_rates[186] - rev_rates[185]) * pres_mod[28]
               - (fwd_rates[187] - rev_rates[186]) - (fwd_rates[188] - rev_rates[187])
               - (fwd_rates[280] - rev_rates[279]) - (fwd_rates[281] - rev_rates[280])
              ;

  sp_rates[37] = -(fwd_rates[180] - rev_rates[179]) - (fwd_rates[181] - rev_rates[180])
               - (fwd_rates[182] - rev_rates[181]) - (fwd_rates[183] - rev_rates[182])
               - (fwd_rates[184] - rev_rates[183]) * pres_mod[27] + (fwd_rates[198] - rev_rates[197])
               + (fwd_rates[227] - rev_rates[226]) + (fwd_rates[281] - rev_rates[280])
              ;

  sp_rates[38] = (fwd_rates[191] - rev_rates[190]) + (fwd_rates[193] - rev_rates[192])
               + (fwd_rates[196] - rev_rates[195]) + (fwd_rates[200] - rev_rates[199])
               + (fwd_rates[211] - rev_rates[210]) * pres_mod[30] - (fwd_rates[212] - rev_rates[211])
               - (fwd_rates[213] - rev_rates[212]) - (fwd_rates[214] - rev_rates[213])
               - (fwd_rates[215] - rev_rates[214]) + (fwd_rates[262] - rev_rates[261])
               + (fwd_rates[279] - rev_rates[278]);

  sp_rates[39] = -(fwd_rates[216] - rev_rates[215]) - (fwd_rates[217] - rev_rates[216])
               - (fwd_rates[218] - rev_rates[217]) - (fwd_rates[219] - rev_rates[218])
               - (fwd_rates[220] - rev_rates[219]) + (fwd_rates[229] - rev_rates[228]) * pres_mod[32]
               + (fwd_rates[232] - rev_rates[231]) + (fwd_rates[238] - rev_rates[237])
               + (fwd_rates[243] - rev_rates[242]) - (fwd_rates[280] - rev_rates[279])
              ;

  sp_rates[40] = (fwd_rates[218] - rev_rates[217]) + (fwd_rates[220] - rev_rates[219])
               - (fwd_rates[229] - rev_rates[228]) * pres_mod[32] - (fwd_rates[230] - rev_rates[229])
               - (fwd_rates[231] - rev_rates[230]) - (fwd_rates[232] - rev_rates[231])
               - (fwd_rates[233] - rev_rates[232]) - (fwd_rates[234] - rev_rates[233])
               - (fwd_rates[235] - rev_rates[234]) - (fwd_rates[236] - rev_rates[235]) * pres_mod[33]
               + (fwd_rates[239] - rev_rates[238]) + (fwd_rates[241] - rev_rates[240])
               + (fwd_rates[242] - rev_rates[241]) + (fwd_rates[245] - rev_rates[244])
               + (fwd_rates[249] - rev_rates[248]) + (fwd_rates[252] - rev_rates[251])
               + (fwd_rates[254] - rev_rates[253]) + (fwd_rates[257] - rev_rates[256])
               + (fwd_rates[270] - rev_rates[269]) + (fwd_rates[275] - rev_rates[274])
              ;

  sp_rates[41] = (fwd_rates[236] - rev_rates[235]) * pres_mod[33] - (fwd_rates[237] - rev_rates[236])
               + (fwd_rates[255] - rev_rates[254]) + (fwd_rates[274] - rev_rates[273])
              ;

  sp_rates[42] = (fwd_rates[240] - rev_rates[239]) * pres_mod[34] - (fwd_rates[256] - rev_rates[255])
               - (fwd_rates[257] - rev_rates[256]) - (fwd_rates[258] - rev_rates[257])
               - (fwd_rates[259] - rev_rates[258]) - (fwd_rates[260] - rev_rates[259])
              ;

  sp_rates[43] = (fwd_rates[250] - rev_rates[249]) + (fwd_rates[253] - rev_rates[252])
               - (fwd_rates[269] - rev_rates[268]) - (fwd_rates[270] - rev_rates[269])
               - (fwd_rates[271] - rev_rates[270]) + (fwd_rates[273] - rev_rates[272])
              ;

  sp_rates[44] = (fwd_rates[233] - rev_rates[232]) - (fwd_rates[272] - rev_rates[271])
              ;

  sp_rates[45] = (fwd_rates[234] - rev_rates[233]) + (fwd_rates[248] - rev_rates[247])
               + (fwd_rates[251] - rev_rates[250]) - (fwd_rates[261] - rev_rates[260])
               - (fwd_rates[262] - rev_rates[261]) - (fwd_rates[263] - rev_rates[262])
               - (fwd_rates[264] - rev_rates[263]) - (fwd_rates[265] - rev_rates[264])
               - (fwd_rates[266] - rev_rates[265]) - (fwd_rates[267] - rev_rates[266])
               - (fwd_rates[268] - rev_rates[267]) * pres_mod[35] + (fwd_rates[269] - rev_rates[268])
               + (fwd_rates[272] - rev_rates[271]);

  sp_rates[46] = (fwd_rates[217] - rev_rates[216]) + (fwd_rates[219] - rev_rates[218])
               - (fwd_rates[221] - rev_rates[220]) - (fwd_rates[222] - rev_rates[221])
               - (fwd_rates[223] - rev_rates[222]) - (fwd_rates[224] - rev_rates[223])
               - (fwd_rates[225] - rev_rates[224]) - (fwd_rates[226] - rev_rates[225]) * pres_mod[31]
               - (fwd_rates[227] - rev_rates[226]) - (fwd_rates[228] - rev_rates[227])
               + (fwd_rates[230] - rev_rates[229]) + (fwd_rates[246] - rev_rates[245])
               + (fwd_rates[263] - rev_rates[262]) + (fwd_rates[265] - rev_rates[264])
               + (fwd_rates[266] - rev_rates[265]) + (fwd_rates[280] - rev_rates[279])
               - (fwd_rates[281] - rev_rates[280]);

  sp_rates[47] = (fwd_rates[177] - rev_rates[176]) + (fwd_rates[180] - rev_rates[179])
               + (fwd_rates[182] - rev_rates[181]) + (fwd_rates[183] - rev_rates[182])
               + (fwd_rates[184] - rev_rates[183]) * pres_mod[27] + (fwd_rates[195] - rev_rates[194])
               + (fwd_rates[197] - rev_rates[196]) + (fwd_rates[203] - rev_rates[202])
               + (fwd_rates[204] - rev_rates[203]) * pres_mod[29] + (fwd_rates[205] - rev_rates[204])
               + (fwd_rates[206] - rev_rates[205]) + (fwd_rates[208] - rev_rates[207])
               + (fwd_rates[209] - rev_rates[208]) + (fwd_rates[210] - rev_rates[209])
               + (fwd_rates[224] - rev_rates[223]) + (fwd_rates[228] - rev_rates[227])
               + (fwd_rates[237] - rev_rates[236]) - (fwd_rates[238] - rev_rates[237])
               - (fwd_rates[239] - rev_rates[238]) - (fwd_rates[240] - rev_rates[239]) * pres_mod[34]
               - (fwd_rates[241] - rev_rates[240]) - (fwd_rates[242] - rev_rates[241])
               + (fwd_rates[256] - rev_rates[255]) + (fwd_rates[258] - rev_rates[257])
               + (fwd_rates[259] - rev_rates[258]) + (fwd_rates[260] - rev_rates[259])
              ;

  sp_rates[48] = 0.0;

  sp_rates[49] = (fwd_rates[312] - rev_rates[297]) + (fwd_rates[313] - rev_rates[298])
               + (fwd_rates[314] - rev_rates[299]) - (fwd_rates[315] - rev_rates[300])
               + (fwd_rates[316] - rev_rates[301]) + (fwd_rates[317] - rev_rates[302]) * pres_mod[39]
               - (fwd_rates[318] - rev_rates[303]) - (fwd_rates[319] - rev_rates[304]) * pres_mod[40]
               - (fwd_rates[320] - rev_rates[305]) - (fwd_rates[321] - rev_rates[306])
               - (fwd_rates[322] - rev_rates[307]) - fwd_rates[323] - (fwd_rates[324] - rev_rates[308]);

  sp_rates[50] = (fwd_rates[311] - rev_rates[296]) * pres_mod[38] - (fwd_rates[312] - rev_rates[297])
               - (fwd_rates[313] - rev_rates[298]) - (fwd_rates[314] - rev_rates[299])
               + (fwd_rates[315] - rev_rates[300]) - (fwd_rates[316] - rev_rates[301])
               + (fwd_rates[319] - rev_rates[304]) * pres_mod[40] + (fwd_rates[322] - rev_rates[307])
              ;

  sp_rates[51] = (fwd_rates[284] - rev_rates[282]) + (fwd_rates[293] - rev_rates[287])
               + (fwd_rates[295] - rev_rates[289]) + (fwd_rates[298] - rev_rates[290])
               + (fwd_rates[303] - rev_rates[291]) * pres_mod[37] - fwd_rates[304] - fwd_rates[305]
               - fwd_rates[306] - (fwd_rates[307] - rev_rates[292]) - (fwd_rates[308] - rev_rates[293])
               - (fwd_rates[309] - rev_rates[294]) - (fwd_rates[310] - rev_rates[295])
              ;

  sp_rates[52] = (fwd_rates[285] - rev_rates[283]) - (fwd_rates[295] - rev_rates[289])
               - fwd_rates[296] - fwd_rates[297] - (fwd_rates[298] - rev_rates[290]) - fwd_rates[299]
               - fwd_rates[300] - fwd_rates[301] - fwd_rates[302];

} // end eval_spec_rates

