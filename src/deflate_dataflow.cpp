// Parallel deflate in Dataflow coding style.
//
//

#include <stdio.h>
#include <string.h>
#include <hls_stream.h>
#include <ap_utils.h>
#include <ap_int.h>

using namespace hls;

// TODO: remove debug print.
#include <iostream>
using namespace std;

#include "constant.h"

#define DIV_CEIL(x, base) (((x) + (base) - 1 ) / (base))

////////////////////////////////////////////////////////////////////////////
//
//  Helper functions to avoid pointer casting related error in HLS.
//
////////////////////////////////////////////////////////////////////////////

void uint512_to_vec_t(vec_t dst[64/VEC], uint512 src) {
  int j;
#pragma HLS inline
  for (j=0; j<64/VEC; j++) {
#pragma HLS UNROLL
    dst[j] = src((j+1)*VEC*8-1, j*VEC*8);
  }
}

void vec_t_to_chars(uint8 dst[VEC], vec_t src) {
  int j;
#pragma HLS inline
  for (j=0; j<VEC; j++) {
#pragma HLS UNROLL
    dst[j] = src((j+1)*8-1, j*8);
  }
}

vec_t chars_to_vec_t(uint8 src[VEC]) {
  int j;
  vec_t ret;
#pragma HLS inline
  for (j=0; j<VEC; j++) {
#pragma HLS UNROLL
    ret((j+1)*8-1, j*8) = src[j];
  }
  return ret;
}

vec_2t uint16_to_vec_2t(uint16 src[VEC]) {
  int j;
  vec_2t ret;
#pragma HLS inline
  for (j=0; j<VEC; j++) {
#pragma HLS UNROLL
    ret((j+1)*16-1, j*16) = src[j];
  }
  return ret;
}

void vec_2t_to_uint16(uint16 dst[VEC], vec_2t src) {
  int j;
#pragma HLS inline
  for (j=0; j<VEC; j++) {
#pragma HLS UNROLL
    dst[j] = src((j+1)*16-1, j*16);
  }
}

uint512 chars_to_uint512(uint8 src[64]) {
  int j;
  uint512 ret;
#pragma HLS inline
  for (j=0; j<64; j++) {
#pragma HLS UNROLL
    ret((j+1)*8-1, j*8) = src[j];
  }
  return ret;
}

void uint64_to_uint16(uint16 dst[4], uint64 src) {
  int j;
#pragma HLS inline
  for (j=0; j<4; j++) {
#pragma HLS UNROLL
    dst[j] = src((j+1)*16-1, j*16);
  }
}

uint64 uint16_to_uint64(uint16 src[4]) {
  int j;
  uint64 ret;
#pragma HLS inline
  for (j=0; j<4; j++) {
#pragma HLS UNROLL
    ret((j+1)*16-1, j*16) = src[j];
  }
  return ret;
}

void vec_8t_to_h(uint16 h[4*VEC], vec_8t src) {
#pragma HLS inline
  int j;
  for (j=0; j<4*VEC;j++) {
#pragma HLS UNROLL
    h[j] = src((j+1)*16-1, j*16);
  }
}

vec_8t h_to_vec_8t(uint16 src[VEC]) {
  int j;
  vec_8t ret;
#pragma HLS inline
  for (j=0; j<4*VEC; j++) {
#pragma HLS UNROLL
    ret((j+1)*16-1, j*16) = src[j];
  }
  return ret;
}

vec_2t chars_to_vec_2t(uint8 src[VEC*2]) {
  int j;
  vec_2t ret;
#pragma HLS inline
  for (j=0; j<2*VEC; j++) {
#pragma HLS UNROLL
    ret((j+1)*8-1, j*8) = src[j];
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////
void write_buf(stream<vec_t> &feed_out, uint512* buf, int vec_count) {
  int i;
  for (i=0; i<DEPTH*(64/VEC); i++) {
#pragma HLS PIPELINE
    uint512 elem = buf[i/(64/VEC)];
    int j = i % (64/VEC);
    vec_t split_buffer = elem((j+1)*(8*VEC)-1, j*(8*VEC));
    feed_out.write(split_buffer);
    if (i == vec_count-1) {
      break;
    }
  }
}

void copy_input_buf(uint512 *in_buf, uint512 *local_buf, int count) {
  memcpy(local_buf, (void*)&in_buf[(count)*DEPTH], DEPTH*64);
}

void feed(uint512 *in_buf, int in_size, stream<vec_t>& feed_out) {
  int batch_count = DIV_CEIL(in_size, 64*DEPTH);
  int vec_count = DIV_CEIL(in_size, VEC);
  int last_vec_count = vec_count%(64*DEPTH/VEC);
  uint512 buf1[DEPTH];
  uint512 buf2[DEPTH];
  if (last_vec_count == 0) {
    last_vec_count = 64*DEPTH/VEC;
  }
  int i;
  for (i=0; i<batch_count+1; i++) {
    if (i % 2 == 0) {
      if (i == 0) {
        copy_input_buf(in_buf, buf1, i);
        //memcpy(buf1, (const void*)&in_buf[i*128], 64*128);
      } else {
        if (i == batch_count) {
          write_buf(feed_out, buf2, last_vec_count);
        } else {
          //memcpy(buf1, (const void*)&in_buf[i*128], 64*128);
          copy_input_buf(in_buf, buf1, i);
          write_buf(feed_out, buf2, 64*DEPTH/VEC);
        }
      }
    } else {
      if (i == batch_count) {
        write_buf(feed_out, buf1, last_vec_count);
      } else {
        //memcpy(buf2, (const void*)&in_buf[i*128], 64*128);
        copy_input_buf(in_buf, buf2, i);
        write_buf(feed_out, buf1, 64*DEPTH/VEC);
      }
    }
  }
}

// Compute hash value from 4 bytes of input.
uint32 compute_hash(vec_t line_in) {
#pragma HLS inline
  uint8 input[VEC];
#pragma HLS ARRAY_PARTITION variable=input complete
  vec_t_to_chars(input, line_in);

  return (((uint32)input[0])<<5) ^
          (((uint32)input[1])<<4) ^
          (((uint32)input[2])<<3) ^
          (((uint32)input[3])<<2) ^
          (((uint32)input[4])<<1) ^
          ((uint32)input[5]);
}

// Compare two strings and calculate the length of match
uint16 calc_match_len(vec_t line_in, vec_t record_in) {
  uint16 match_id[VEC];
  uint8 input[VEC];
  uint8 record[VEC];
#pragma HLS inline
#pragma HLS ARRAY_PARTITION variable=match_id complete
#pragma HLS ARRAY_PARTITION variable=input complete
#pragma HLS ARRAY_PARTITION variable=record complete
  uint16 i;
  vec_t_to_chars(input, line_in);
  vec_t_to_chars(record, record_in);
  for (i=0; i<VEC; i++) {
    if (input[i] == record[i]) match_id[i] = VEC;
    else match_id[i] = i;
  }
  uint16 min_v, min_i;
  min_reduction(match_id, &min_v, &min_i);
  if (min_v == VEC) {
    min_i = VEC;
  }
  return min_i;
}

void gather_match_results(vec_t match_candidates[HASH_TABLE_BANKS],
    uint32 match_positions_c[HASH_TABLE_BANKS],
    uint8 match_valid_c[HASH_TABLE_BANKS],
    uint16 bank_num1[VEC], vec_t match_results[VEC],
    uint32 match_positions[VEC], uint8 match_valid[VEC]) {
  // Gather match results from correct banks.
  // Now match_result[i][j] is the potential match for string current[j] in
  // dictionaty i.
  int j,k;
  vec_t match_results_regs[HASH_TABLE_BANKS][VEC];
#pragma HLS ARRAY_PARTITION variable=match_results_regs complete
  uint32 match_positions_regs[HASH_TABLE_BANKS][VEC];
#pragma HLS ARRAY_PARTITION variable=match_positions_regs complete
  uint8 match_valid_regs[HASH_TABLE_BANKS][VEC];
#pragma HLS ARRAY_PARTITION variable=match_valid_regs complete

  for (k = 0; k < HASH_TABLE_BANKS; k++) {
    for (j=0; j<VEC; j++) {
      if (bank_num1[j] == k) {
        match_results_regs[k][j] = match_candidates[k];
        match_positions_regs[k][j] = match_positions_c[k];
        match_valid_regs[k][j] = match_valid_c[k];
      } else {
        if (k != 0) {
          match_results_regs[k][j] = match_results_regs[k-1][j];
          match_positions_regs[k][j] = match_positions_regs[k-1][j];
          match_valid_regs[k][j] = match_valid_regs[k-1][j];
        } else {
          match_results_regs[k][j] = 0;
          match_positions_regs[k][j] = 0;
          match_valid_regs[k][j] = 0;
        }
      }
    }
  }
  for (j=0; j<VEC; j++) {
    match_results[j] = match_results_regs[HASH_TABLE_BANKS-1][j];
    match_positions[j] = match_positions_regs[HASH_TABLE_BANKS-1][j];
    match_valid[j] = match_valid_regs[HASH_TABLE_BANKS-1][j];
  }
}

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

void match_selection(stream<vec_t> &literals, stream<vec_2t> &len_raw,
     stream<vec_2t> &dist_raw, int in_size, stream<vec_2t> &len,
     stream<vec_2t> &dist, stream<vec_t> &valid) {
  int i, j, k;
  int vec_batch_count = DIV_CEIL(in_size, VEC);

  // Loop carried match head position
  uint16 head_match_pos = 0;

  for (i = 0; i < vec_batch_count; i++) {
#pragma HLS PIPELINE
    uint16 reach[VEC];
#pragma HLS ARRAY_PARTITION variable=reach complete
    uint16 larray[VEC];
#pragma HLS ARRAY_PARTITION variable=larray complete
    uint16 darray[VEC];
#pragma HLS ARRAY_PARTITION variable=darray complete
    uint8 ldvalid[VEC];
#pragma HLS ARRAY_PARTITION variable=ldvalid complete
    uint8 literals_vec[VEC];
#pragma HLS ARRAY_PARTITION variable=literals_vec complete
    vec_2t len_raw_read, dist_raw_read, len_write, dist_write;
    vec_t literals_read;

    if (len_raw.empty() || dist_raw.empty() || literals.empty()) {
      i--;
      continue;
    }

    len_raw.read(len_raw_read);
    dist_raw.read(dist_raw_read);
    literals.read(literals_read);
    vec_2t_to_uint16(larray, len_raw_read);
    vec_2t_to_uint16(darray, dist_raw_read);
    vec_t_to_chars(literals_vec, literals_read);

    // First we compute how far each match / literal can reach
    for (j=0; j<VEC; j++) {
      if (darray[j] != 0) {
        // cannot exceed input_pos + 2 * VEC - 1: within next window.
        reach[j] = j + larray[j] + 3;
      } else {
        reach[j] = j + 1;
      }
    }

    // reach_1[k][j] is the reach[j] given the previous match covers
    // until index k. We essentialy speculate the max reach for
    // all cases.

    uint16 max_reach, max_reach_index;
    max_reduction(reach, &max_reach, &max_reach_index);

    // First determine the position of head for the next cycle.
    uint16 old_head_match_pos = head_match_pos;
    // If the max reach position is VEC+1 then the max rreach index must be
    // smaller than or equal to VEC-2. In that case a head match = VEC-1 will
    // demote this match to 2 characters. In that case we use the last character
    // instead.
    if (max_reach == VEC + 1 && old_head_match_pos == VEC - 1) {
      head_match_pos = 0;
    } else {
      head_match_pos = max_reach - VEC;
    }
/*
    cout << "i: " << i*VEC << " HEAD=" << old_head_match_pos << " max_reach="
         << max_reach << " max_reach_index=" << max_reach_index << endl;
    for (j=0; j<VEC; j++) {
      cout << "  (L,D): " << larray[j] << " " << darray[j]
           << " -- " << hex << literals_vec[j] << dec << endl;
    }
*/
    // Perform tail match trimming if overlapping with head.
    // First is the special case of demotion. In this case the last
    // (L D) must have D=0.
    if (max_reach == VEC + 1 && old_head_match_pos == VEC - 1) {
      max_reach_index = VEC - 1;
    } else if ((max_reach == VEC && old_head_match_pos == VEC - 2)
        || (max_reach == VEC && old_head_match_pos == VEC - 1)) {
      max_reach_index = VEC - 1;
    } else if (max_reach_index < old_head_match_pos) {
      uint16 new_max_reach_index = old_head_match_pos;
      uint16 diff = new_max_reach_index - max_reach_index;
      larray[new_max_reach_index] = larray[max_reach_index] - diff;
      darray[new_max_reach_index] = darray[max_reach_index];  //WK: no need to add diff, since the last dictionary item is the same
      reach[new_max_reach_index] = reach[max_reach_index];
      max_reach_index = new_max_reach_index;
    }

    // Perform pipelined match selection: set valid bit for all matches.
    // Pass 1: Eliminate any matches that (1) extend beyond last match starting
    // position; (2) already handled by last cycle; (3) already handled by
    // the last match. Can be performed in parallel.
    for (k=0; k<VEC; k++) {
      if (k < old_head_match_pos || k > max_reach_index) {
        ldvalid[k] = 0;
      } else if (k == max_reach_index) {
        ldvalid[k] = 1; // Last match / literal must be valid
      } else {
        // Trim a match if it overlaps with the final match
        if (darray[k] != 0 && reach[k] > max_reach_index) {
          uint16 trimmed_len = larray[k] + 3 + max_reach_index - reach[k];
          if (trimmed_len < 3) {
            larray[k] = literals_vec[k];
            darray[k] = 0;
          } else {
            larray[k] = trimmed_len - 3;
          }
        }
        ldvalid[k] = 1;
      }
    }

    uint16 processed_len = old_head_match_pos;
    // Pass 2: For all the remaining matches, filter with lazy evaluztion.
    for (k=0; k<VEC-1; k++) {
      // Make sure we don't touch the tail match 
      if (ldvalid[k] && k != max_reach_index) {
        if (k < processed_len) {
          ldvalid[k] = 0;
        } else if (darray[k] == 0) { // literal should be written out
          processed_len++;
        } else { // current position is a match candidate
          // When the next match is better: commit literal here instead of match
          if (ldvalid[k+1] && darray[k+1] > 0 && larray[k+1] > larray[k]) {
            larray[k] = literals_vec[k];
            darray[k] = 0;
            processed_len++;
          } else {
            processed_len += larray[k] + 3;
          }
        }
      }
    }
/*
    for (j=0; j<VEC; j++) {
      cout << "  *(L,D): " << larray[j] << " " << darray[j]
           << " -- " << ldvalid[j] << endl;
    }
*/
    len.write(uint16_to_vec_2t(larray));
    dist.write(uint16_to_vec_2t(darray));
    valid.write(chars_to_vec_t(ldvalid));
  }
}

void mock_lz77(stream<vec_t> &literals, stream<vec_2t> &len,
    stream<vec_2t> &dist, stream<vec_t> &valid, int in_size) {
  int i, j;
  uint8 char_in[VEC];
  uint16 short_out[VEC];
  uint8 valid_out[VEC];
  int vec_batch_count = DIV_CEIL(in_size, VEC);
  for (i = 0; i < vec_batch_count; i++) {
#pragma HLS PIPELINE
    vec_t literals_read;
    if (literals.empty()) {
      i--;
      continue;
    }
    literals.read(literals_read);
    vec_t_to_chars(char_in, literals_read);
    for (j = 0; j < VEC; j++) {
      short_out[j] = char_in[j];
      valid_out[j] = 1;
    }
    len.write(uint16_to_vec_2t(short_out));
    dist.write(0);
    valid.write(chars_to_vec_t(valid_out));
  }
}

void parallel_huffman_encode(stream<vec_2t> &len, stream<vec_2t> &dist,
    stream<vec_t> &valid, int in_size, stream<uint16> &total_len,
    stream<vec_8t> &hcode8, stream<vec_8t> &hlen8) {
  int i, j;

  //cout << "Started huffman. " << endl;
  // Injects end code 256.
  int vec_batch_count = DIV_CEIL(in_size+1, VEC);

  int counter = 0;

  for (i = 0; i < vec_batch_count; i++) {
#pragma HLS pipeline
    uint64 hcode[VEC];
    uint64 hlen[VEC];
    uint16 len_vec[VEC];
    uint16 dist_vec[VEC];
    uint16 total_len_vec[VEC];
    uint8 valid_vec[VEC];
    vec_2t len_current, dist_current;
    vec_t valid_current;

    // There will be one more output line if the original size is
    //   a multiple of VEC. Simple add an empty line.
    if (in_size % VEC == 0 && i == vec_batch_count - 1) {
      len_current = 0;
      dist_current = 0;
      valid_current = 0;
    } else {
      if (len.empty() || dist.empty() || valid.empty()) {
        i--;
        continue;
      }
      len.read(len_current);
      dist.read(dist_current);
      valid.read(valid_current);
    }
    vec_2t_to_uint16(len_vec, len_current);
    vec_2t_to_uint16(dist_vec, dist_current);
    vec_t_to_chars(valid_vec, valid_current);

    for (j = 0; j < VEC; j++) {
      if (i * VEC + j == in_size) {
        len_vec[j] = 256;
        dist_vec[j] = 0;
        valid_vec[j] = 1;
      } else if (i * VEC + j > in_size) {
        len_vec[j] = 0;
        dist_vec[j] = 0;
        valid_vec[j] = 0;
      }
      huffman_translate(len_vec[j], dist_vec[j],
          &total_len_vec[j], &hcode[j], &hlen[j]);
      if (valid_vec[j] == 0) {
        hcode[j] = 0;
        hlen[j] = 0;
        total_len_vec[j] = 0;
      }
    }

    uint16 total_len_out = 0;
    for (j = 0; j < VEC; j++) {
      total_len_out += total_len_vec[j];
    }

    counter += total_len_out;
//    cout << "Total len: " << counter << "; ("
//         << DIV_CEIL(counter, 8) << ")" << endl;

    vec_8t hcode8_out, hlen8_out;
    for (j = 0; j < VEC; j++) {
      hcode8_out((j+1)*64-1, j*64) = hcode[j];
      hlen8_out((j+1)*64-1, j*64) = hlen[j];
    }
    hcode8.write(hcode8_out); hlen8.write(hlen8_out);
    total_len.write(total_len_out);
  }
}

void huffman_local_pack(vec_8t hcode_in, vec_8t hlen_in, int old_bit_pos, int out_count,
    vec_2t *work_buf_out, vec_2t *work_buf_next_out) {
#pragma HLS INLINE
  // Starting bit position of each code.
  uint32 pos[4*VEC];
  // Starting short position of each code.
  uint32 s_pos[4*VEC];

  uint32 hcode[4*VEC];
  uint32 hlen[4*VEC];
  uint16 zero16 = 0;

  int j, k;
  vec_2t work_buf = 0, work_buf_next = 0;

  // Compressed and packed data -- ready to send to output. Use double buffering
  // to reduce stall.
  // We know lcode, lextra, dcode, dextra's huffman codes are no more than 16 
  // bits, therefore it's impossible in one iteration to use up both buffers.
  uint16 local_pack[VEC];
#pragma HLS ARRAY_PARTITION variable=local_pack complete
  uint16 local_pack_next[VEC];
#pragma HLS ARRAY_PARTITION variable=local_pack_next complete
  for (j = 0; j < 4*VEC; j++) {
    hcode[j] = hcode_in((j+1)*16-1, j*16);
    hlen[j] = hlen_in((j+1)*16-1, j*16);
  }

  // Huffman packing.

  pos[0] = 0;
  for (j = 1; j < 4*VEC; j++) {
    pos[j] = pos[j-1] + hlen[j-1];
  }

  vec_2t pack_temp;
  vec_4t pack_shifted;
  pack_temp = 0;
#define PRESHIFT
#ifdef PRESHIFT
  vec_t2 local_pack_temp[4];
  for (j = 0; j < 4; j++) {
    local_pack_temp[j] = 0;
  }

  for (j = 0; j < VEC; j++) {
    for (k = 0; k < 4; k++) {
      vec_t2 temp1 = hcode[j + k * VEC];
      uint16 offset = pos[j+VEC*k] - pos[VEC*k];
      local_pack_temp[k] |= temp1 << offset;
    }
  }

  for (j = 0; j < 4; j++) {
    vec_2t temp = local_pack_temp[j];
    pack_temp |= temp << pos[j * VEC];
  }

#else
  for (j=0; j<VEC*4; j++) {
    vec_2t temp = hcode[j];
    pack_temp |= temp << pos[j];
  }
#endif

  pack_shifted = pack_temp;
  pack_shifted = pack_shifted << (old_bit_pos % (VEC*16));
  work_buf = pack_shifted(VEC*16-1, 0);
  work_buf_next = pack_shifted(VEC*32-1, VEC*16);

  *work_buf_out = work_buf;
  *work_buf_next_out = work_buf_next;
}

void write_huffman_output(stream<uint16> &total_len,
    stream<vec_8t> &hcode8, stream<vec_8t> &hlen8,
    int in_size, stream<vec_2t> &data, stream<int> &size) {
  // Input huffman tree size in bytes

//  int tree_size_bits_fixed = 640;
  int tree_size_bits_fixed = 3;
  int i, j, out_count;

  // The number of input batches
  int in_batch_count = DIV_CEIL(in_size+1, VEC);

  int bit_pos = 0;
  vec_2t partial_out;

  // Record the number of 2*VEC byte buffers wrote to out_buf.
  out_count = 0;

  partial_out = uint512("3");
  out_count = 0;
  bit_pos = tree_size_bits_fixed;

  for (i = 0; i < in_batch_count; i++) {
#pragma HLS PIPELINE II=1
    vec_8t hcode8_out, hlen8_out;
    vec_2t work_buf, work_buf_next;

    if (total_len.empty() ||
        hcode8.empty() || hlen8.empty()) {
      i--;
      continue;
    }

    // Load data from channels
    hcode8.read(hcode8_out); hlen8.read(hlen8_out);
    uint16 total_len_current;
    total_len.read(total_len_current);

    huffman_local_pack(hcode8_out, hlen8_out, bit_pos, out_count,
        &work_buf, &work_buf_next);

    // bit pos is the register to carry through the loop so we update immediately
    int new_bit_pos = (int)total_len_current + bit_pos;
    bool reached_next = (new_bit_pos / (VEC*16) > out_count);
    bit_pos = new_bit_pos;

    if (reached_next) {
      out_count++;
    }

    if (reached_next) {
      data.write(work_buf | partial_out);
      size.write(-1);
      partial_out = work_buf_next;
    } else {
      partial_out |= work_buf;
    }
  }

  // Flush partial_out
  if (bit_pos % (VEC*16) != 0) {
    data.write(partial_out);
    size.write(-1);
  }

  data.write(0);
  size.write(DIV_CEIL(bit_pos, 8));
}

int fill_buf(stream<vec_2t> &data, stream<int> &size, uint512 *local_buf) {
  int count, out_count = 0;
  uint512 tmp_out = 0;
  for (count=0; count<DEPTH*(32/VEC); count++) {
#pragma HLS pipeline
    if (data.empty() || size.empty()) {
      count--;
      continue;
    }
    vec_2t data_read;
    int size_read;
    data.read(data_read);
    size.read(size_read);
    if (size_read == -1) {
      int slot = count % (32/VEC);
      tmp_out((slot+1)*VEC*16-1, slot*VEC*16) = data_read;
      //count++;
      if (slot == (32/VEC-1)) {
        local_buf[out_count] = tmp_out;
        out_count++;
        tmp_out = 0;
      }
    } else {
      // Flush uint512 buffer.
      if (count % (32/VEC) != 0) {
        local_buf[out_count] = tmp_out;
      }
      return size_read;
    }
  }
  return -1;
}

void copy_local_buf(uint512 *out_buf, uint512 *local_buf, int count) {
  memcpy((void*)&out_buf[(count)*DEPTH], local_buf, DEPTH*64);
}

void export_data(stream<vec_2t> &data, stream<int> &size, uint512 *out_buf,
    int *out_size) {
  uint32 buf = 0;
  uint512 buf1[DEPTH];
  uint512 buf2[DEPTH];
  int out_size_local;
  while(1) {
    if (buf % 2 == 0) {
      if (buf == 0) {
        out_size_local = fill_buf(data, size, buf1);
      } else {
        if (out_size_local != -1) {
          if (out_size_local % (DEPTH*64) != 0) {
            copy_local_buf(out_buf, buf2, buf-1);
            //memcpy((void*)&out_buf[(buf-1)*128], buf2, 128*64);
          }
          *out_size = out_size_local;
          break;
        } else {
          out_size_local = fill_buf(data, size, buf1);
          copy_local_buf(out_buf, buf2, buf-1);
          //memcpy((void*)&out_buf[(buf-1)*128], buf2, 128*64);
        }
      }
    } else {
        if (out_size_local != -1) {
          if (out_size_local % (DEPTH*64) != 0) {
            //memcpy((void*)&out_buf[(buf-1)*128], buf1, 128*64);
            copy_local_buf(out_buf, buf1, buf-1);
          }
          *out_size = out_size_local;
          break;
        } else {
          out_size_local = fill_buf(data, size, buf2);
          //memcpy((void*)&out_buf[(buf-1)*128], buf1, 128*64);
          copy_local_buf(out_buf, buf1, buf-1);
        }
    }
    buf++;
  }
}

void no_comp(int in_size, stream<vec_t> &literals,
     stream<vec_2t> &data, stream<int> &size) {
  int vec_batch_count = DIV_CEIL(in_size, VEC);
  int i;
  vec_2t buf;
  for (i=0; i<vec_batch_count; i++) {
#pragma HLS pipeline
    if (literals.empty()) {
      i--;
      continue;
    }
    vec_t literals_read;
    literals.read(literals_read);
    int j = i%2;
    buf(VEC*8*(j+1)-1, VEC*8*j) = literals_read;
    if (j == 1) {
      data.write(buf);
      size.write(-1);
    }
  }
  if (vec_batch_count % 2 == 1) {
    data.write(buf);
    size.write(-1);
  }
  data.write(0);
  size.write(in_size);
}

extern "C" {
void deflate(uint512 *in_buf, int in_size,
                uint512 *out_buf, int *out_size) {
#pragma HLS INTERFACE m_axi port=in_buf offset=slave bundle=gmem0 depth=1024
#pragma HLS INTERFACE m_axi port=out_buf offset=slave bundle=gmem1 depth=2048
#pragma HLS INTERFACE m_axi port=out_size offset=slave bundle=gmem1 depth=1
//#pragma HLS INTERFACE s_axilite port=in_size bundle=control
//#pragma HLS INTERFACE s_axilite port=tree_size bundle=control
//#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=in_buf bundle=control
#pragma HLS INTERFACE s_axilite port=in_size bundle=control
#pragma HLS INTERFACE s_axilite port=out_buf bundle=control
#pragma HLS INTERFACE s_axilite port=out_size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  stream<vec_t> literals("literals");
#pragma HLS STREAM variable=literals depth=2048

  stream<vec_2t> current_window("current_window");
#pragma HLS STREAM variable=current_window depth=32
  stream<vec_2t> bank_num1("bank_num1");
#pragma HLS STREAM variable=bank_num1 depth=32  
  stream<vec_2t> bank_occupied("bank_occupied");
#pragma HLS STREAM variable=bank_occupied depth=32
  stream<vec_2t> bank_offset("bank_offset");
#pragma HLS STREAM variable=bank_offset depth=32

  stream<vec_t> literals_2("literals_2");
#pragma HLS STREAM variable=literals_2 depth=2048
  stream<vec_2t> len_raw("len_raw");
#pragma HLS STREAM variable=len_raw depth=2048
  stream<vec_2t> dist_raw("dist_raw");
#pragma HLS STREAM variable=dist_raw depth=2048

  stream<vec_2t> len("len");
#pragma HLS STREAM variable=len depth=2048
  stream<vec_2t> dist("dist");
#pragma HLS STREAM variable=dist depth=2048
  stream<vec_t> valid("valid");
#pragma HLS STREAM variable=valid depth=2048

  stream<vec_8t> hcode8("hcode8");
#pragma HLS STREAM variable=hcode8 depth=2048
  stream<vec_8t> hlen8("hlen8");
#pragma HLS STREAM variable=hlen8 depth=2048

  stream<vec_2t> data("data");
#pragma HLS STREAM variable=data depth=2048
  stream<int> size("size");
#pragma HLS STREAM variable=size depth=2048
  stream<uint16> total_len("total_len");
#pragma HLS STREAM variable=total_len depth=2048

#pragma HLS DATAFLOW
  int in_size_local = in_size;

  feed(in_buf, in_size_local, literals);

 hash_match_p1(literals, in_size_local, current_window, bank_num1,
      bank_occupied, bank_offset);

  hash_match_p2(current_window, bank_num1, bank_occupied, bank_offset, in_size_local,
      literals_2, len_raw, dist_raw);

  match_selection(literals_2, len_raw, dist_raw, in_size_local,
      len, dist, valid);

  parallel_huffman_encode(len, dist, valid, in_size_local, total_len,
      hcode8, hlen8);

  write_huffman_output(total_len,
      hcode8, hlen8, in_size_local, data, size);

  export_data(data, size, out_buf, out_size);
}
}
