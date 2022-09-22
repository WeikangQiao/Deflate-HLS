void hash_match_p1(stream<vec_t> &data_window, int in_size,
    stream<vec_2t> &current_window_p,
    stream<vec_2t> &bank_num1_p,
    stream<vec_2t> &bank_occupied_p,
    stream<vec_2t> &bank_offset_p) {

  int i, j, k;
  int batch_count = DIV_CEIL(in_size, VEC);
  vec_t data0 = 0, data1 = 0;


  for (i = 0; i < batch_count + 1; i++) {
#pragma HLS PIPELINE

    uint16 hash_v[VEC];
#pragma HLS ARRAY_PARTITION variable=hash_v complete
    uint16 bank_num[VEC];
#pragma HLS ARRAY_PARTITION variable=bank_num complete
    uint16 bank_num1[VEC];
#pragma HLS ARRAY_PARTITION variable=bank_num1 complete
    uint16 bank_occupied[HASH_TABLE_BANKS];
#pragma HLS ARRAY_PARTITION variable=bank_occupied complete
    uint16 bank_offset[HASH_TABLE_BANKS];
#pragma HLS ARRAY_PARTITION variable=bank_offset complete


    if (i < batch_count) {
      if (data_window.empty()) {
        i--;
        continue;
      }
      data0 = data1;
      data_window.read(data1);
    } else {
      data0 = data1;
      data1 = 0;
    }
    vec_t literals_out = data0;

    // Perform hash lookup and calculate match length
    if (i != 0) {
      uint32 input_pos = (i-1) * VEC;
      uint8 data_window0[VEC];
      uint8 data_window1[VEC];

      // For each substring in the window do parallel matching and 
      // LZ77 encoding
      uint16 l[VEC];
      uint16 d[VEC];
      vec_t_to_chars(data_window0, data0);
      vec_t_to_chars(data_window1, data1);

      for (j=0; j<HASH_TABLE_BANKS; j++) {
        bank_offset[j] = BANK_OFFSETS;
        bank_occupied[j] = VEC;
      }
      // Compute hash values for all entries
//      cout  << "BN:";

      vec_t current_vec[VEC];
      vec_2t current_window;
#pragma HLS ARRAY_PARTITION variable=current_vec complete
      uint8 glue[2*VEC];
#pragma HLS ARRAY_PARTITION variable=glue complete
      for (j=0; j<VEC; j++) {
#pragma HLS UNROLL
        glue[j] = data_window0[j];
        glue[j+VEC] = data_window1[j];

        current_window((j+1)*8-1, j*8) = data_window0[j];
        current_window((j+VEC+1)*8-1, (j+VEC)*8) = data_window1[j];
      }


      for (k = 0; k < VEC; k++) {
        current_vec[k] = chars_to_vec_t(&glue[k]);
        hash_v[k] = compute_hash(current_vec[k]);
        //bank_num1[k] = k;
        bank_num1[k] = hash_v[k] % HASH_TABLE_BANKS;
        // bank_num1[k] = hash_v[k] / BANK_OFFSETS % HASH_TABLE_BANKS; // high bits as bank
        bank_num[k] = bank_num1[k];
      }

      // Analyze bank conflict profile and avoid conflicted access
      for (k=0; k<VEC; k++) {
        for (j=0; j<VEC; j++) {
          if (j > k && bank_num[j] == bank_num[k]) {
            bank_num[j] = HASH_TABLE_BANKS; // set to invalid bank number
          }
        }
      }

      // For each bank record its reader/writer.
      for (j=0; j<HASH_TABLE_BANKS; j++) {
        for (k=0; k<VEC; k++) {
          if (bank_num[k] == j) {
            bank_offset[j] = hash_v[k] / HASH_TABLE_BANKS % BANK_OFFSETS;
            //bank_offset[j] = hash_v[k] % BANK_OFFSETS;
            bank_occupied[j] = k;
          }
        }
      }

    // need to pass: current_window, bank_occupied, bank_offset
    current_window_p.write(current_window);
    bank_num1_p.write(uint16_to_vec_2t(bank_num1));
    bank_occupied_p.write(uint16_to_vec_2t(bank_occupied));
    bank_offset_p.write(uint16_to_vec_2t(bank_offset));
    }
  }
}
