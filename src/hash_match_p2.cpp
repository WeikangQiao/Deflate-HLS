void hash_match_p2(stream<vec_2t> &current_window_p,
    stream<vec_2t> &bank_num1_p,
    stream<vec_2t> &bank_occupied_p,
    stream<vec_2t> &bank_offset_p, 
    int in_size,
    stream<vec_t> &literals,
    stream<vec_2t> &len_raw, 
    stream<vec_2t> &dist_raw) 
{
    vec_t hash_content[BANK_OFFSETS][HASH_TABLE_BANKS];
    uint32 hash_position[BANK_OFFSETS][HASH_TABLE_BANKS];
    uint8 hash_valid[BANK_OFFSETS][HASH_TABLE_BANKS];
    vec_t prev_hash_content[HASH_TABLE_BANKS];
    uint32 prev_hash_position[HASH_TABLE_BANKS];
    uint8 prev_hash_valid[HASH_TABLE_BANKS];
    uint32 prev_offset[HASH_TABLE_BANKS];
#pragma HLS ARRAY_PARTITION variable=hash_content complete dim=2
#pragma HLS ARRAY_PARTITION variable=hash_position complete dim=2
#pragma HLS ARRAY_PARTITION variable=hash_valid complete dim=2
#pragma HLS ARRAY_PARTITION variable=prev_hash_content complete dim=1
#pragma HLS ARRAY_PARTITION variable=prev_hash_position complete dim=1
#pragma HLS ARRAY_PARTITION variable=prev_hash_valid complete dim=1
#pragma HLS ARRAY_PARTITION variable=prev_offset complete dim=1

    int i, j, k;
    int vec_batch_count = DIV_CEIL(in_size, VEC);

    // Reset hash table
    for (i=0; i<BANK_OFFSETS; i++) {
    #pragma HLS PIPELINE
        for (j=0; j<HASH_TABLE_BANKS; j++) {
        hash_valid[i][j] = 0;
        }
    }
    for (i = 0; i < HASH_TABLE_BANKS; i++) {
    #pragma HLS UNROLL
        prev_hash_valid[i] = 0;
    }

    for (i = 0; i < vec_batch_count; i++) {
#pragma HLS PIPELINE
    vec_t current_vec[VEC];
#pragma HLS ARRAY_PARTITION variable=current_vec complete
    uint16 bank_num1[VEC];
#pragma HLS ARRAY_PARTITION variable=bank_num1 complete
    uint16 bank_occupied[HASH_TABLE_BANKS];
#pragma HLS ARRAY_PARTITION variable=bank_occupied complete
    uint16 bank_offset[HASH_TABLE_BANKS];
#pragma HLS ARRAY_PARTITION variable=bank_offset complete
    vec_t updates[HASH_TABLE_BANKS];
#pragma HLS ARRAY_PARTITION variable=updates complete
    vec_t match_results[VEC];
#pragma HLS ARRAY_PARTITION variable=match_results complete
    uint32 match_positions[VEC];
#pragma HLS ARRAY_PARTITION variable=match_positions complete
    uint8 match_valid[VEC];
#pragma HLS ARRAY_PARTITION variable=match_valid complete
    vec_t match_candidates[HASH_TABLE_BANKS];
#pragma HLS ARRAY_PARTITION variable=match_candidates complete
    uint32 match_positions_c[HASH_TABLE_BANKS];
#pragma HLS ARRAY_PARTITION variable=match_positions_c complete
    uint8 match_valid_c[HASH_TABLE_BANKS];
#pragma HLS ARRAY_PARTITION variable=match_valid_c complete
#pragma HLS dependence variable=hash_content inter RAW false
#pragma HLS dependence variable=hash_position inter RAW false
#pragma HLS dependence variable=hash_valid inter RAW false

        vec_2t current_window_read, bank_num1_read, bank_occupied_read, bank_offset_read;
        vec_t literals_out;


        if (current_window_p.empty() || bank_num1_p.empty() || bank_occupied_p.empty() || bank_offset_p.empty()) {
            i--;
            continue;
        }

        current_window_p.read(current_window_read);
        bank_num1_p.read(bank_num1_read);
        bank_occupied_p.read(bank_occupied_read);
        bank_offset_p.read(bank_offset_read);

        for (j = 0; j < VEC; j++) {
            current_vec[j](VEC*8-1, 0) = current_window_read((j+VEC)*8-1 , j*8);
        }
        vec_2t_to_uint16(bank_num1, bank_num1_read);
        vec_2t_to_uint16(bank_occupied, bank_occupied_read);
        vec_2t_to_uint16(bank_offset, bank_offset_read);

        literals_out = current_window_read(VEC*8-1, 0);

        // For each substring in the window do parallel matching and 
        // LZ77 encoding
        uint16 l[VEC];
        uint16 d[VEC];

        uint32 input_pos = i * VEC;

        // Prepare update line for the hash table
        for (k=0; k<HASH_TABLE_BANKS; k++) {
            // bank k occupied by string from position writer_pos
            uint8 writer_pos = bank_occupied[k];
            if (writer_pos != VEC) {
            updates[k] = current_vec[writer_pos];
            } else {
            updates[k] = 0;
            }
        }

        // Perform conflict free memory access from all the banks
        for (j=0; j<HASH_TABLE_BANKS; j++) {
            uint8 pos = bank_occupied[j];
            if (pos != VEC) {
            uint32 loffset = bank_offset[j];

            if (loffset != prev_offset[j]) {
                match_candidates[j] = hash_content[bank_offset[j]][j];
                match_positions_c[j] = hash_position[bank_offset[j]][j];
                match_valid_c[j] = hash_valid[bank_offset[j]][j];
            } else {
                match_candidates[j] = prev_hash_content[j];
                match_positions_c[j] = prev_hash_position[j];
                match_valid_c[j] = prev_hash_valid[j];
            }
            } else {
            match_valid_c[j] = 0;
            }
        }

        // Perform hash table update
        for (j=0; j<HASH_TABLE_BANKS; j++) {
            uint8 pos = bank_occupied[j];
            if (pos != VEC) {
            hash_content[bank_offset[j]][j] = updates[j];
            hash_position[bank_offset[j]][j] = input_pos + (uint32)pos;
            hash_valid[bank_offset[j]][j] = 1;

            prev_offset[j] = bank_offset[j];
            prev_hash_content[j] = updates[j];
            prev_hash_position[j] = input_pos + (uint32)pos;
            prev_hash_valid[j] = 1;
            }
        }

        // Gather match results from correct banks.
        // Now match_result[i][j] is the potential match for string current[j] in
        // dictionaty i.
        for (j=0; j<VEC; j++) {
            uint8 k = bank_num1[j];
            if (k != VEC) {
            match_results[j] = match_candidates[k];
            match_positions[j] = match_positions_c[k];
            match_valid[j] = match_valid_c[k];
            } else {
            match_valid[j] = 0;
            }
        }

        for (k=0; k<VEC; k++) {
            uint8 mismatch;
            uint16 ltemp1 = 0;
            ltemp1 = calc_match_len(current_vec[k], match_results[k]);
            uint32 dist = input_pos + k - match_positions[k];
            if (input_pos+k+VEC-1 < in_size && match_valid[k] && ltemp1 >= 3
                && dist <= MAX_MATCH_DIST) {
                l[k] = ltemp1 - 3;
                d[k] = dist;
            } else {
                l[k] = current_window_read((k+1)*8-1, k*8);
                d[k] = 0;
            }
        }

        literals.write(literals_out);
        len_raw.write(uint16_to_vec_2t(l));
        dist_raw.write(uint16_to_vec_2t(d));
    }
}
