/* huffman_translate.c
 *
 * Here we build a function to translate (literal, distance) or (L, D)
 * pair into (huffman code, bit length) pair.
 *
 * The first step is to convert (L, D) into (Lcode, Lextra, Dcode, Dextra)
 * tuple. We can perform this based on the table below, taken from RFC
 * 1951 directly.
 *
 * Then we perform huffman lookup.We use a domain specific huffman tree, 
 * generated from experiment SAM file format. 
 *
 * The actual C code for the tree is generated. Actually to make the code
 * HLS friendly, we prefer manually entered or generated, precomputed 
 * static data.
 */


/*
Deutsch                      Informational                     [Page 11]

RFC 1951      DEFLATE Compressed Data Format Specification      May 1996


                 Extra               Extra               Extra
            Code Bits Length(s) Code Bits Lengths   Code Bits Length(s)
            ---- ---- ------     ---- ---- -------   ---- ---- -------
             257   0     3       267   1   15,16     277   4   67-82
             258   0     4       268   1   17,18     278   4   83-98
             259   0     5       269   2   19-22     279   4   99-114
             260   0     6       270   2   23-26     280   4  115-130
             261   0     7       271   2   27-30     281   5  131-162
             262   0     8       272   2   31-34     282   5  163-194
             263   0     9       273   3   35-42     283   5  195-226
             264   0    10       274   3   43-50     284   5  227-257
             265   1  11,12      275   3   51-58     285   0    258
             266   1  13,14      276   3   59-66

         The extra bits should be interpreted as a machine integer
         stored with the most-significant bit first, e.g., bits 1110
         represent the value 14.

                  Extra           Extra               Extra
             Code Bits Dist  Code Bits   Dist     Code Bits Distance
             ---- ---- ----  ---- ----  ------    ---- ---- --------
               0   0    1     10   4     33-48    20    9   1025-1536
               1   0    2     11   4     49-64    21    9   1537-2048
               2   0    3     12   5     65-96    22   10   2049-3072
               3   0    4     13   5     97-128   23   10   3073-4096
               4   1   5,6    14   6    129-192   24   11   4097-6144
               5   1   7,8    15   6    193-256   25   11   6145-8192
               6   2   9-12   16   7    257-384   26   12  8193-12288
               7   2  13-16   17   7    385-512   27   12 12289-16384
               8   3  17-24   18   8    513-768   28   13 16385-24576
               9   3  25-32   19   8   769-1024   29   13 24577-32768
*/

#include "constant.h"
#include <iostream>
using namespace std;
#define L_EXTRA 29 // from 257 to 285
#define D_EXTRA 30
#define LMAX 286
#define DMAX 30

// Translate literal, distance pair into corresponding huffman encoding and bit count.
// If distance is non zero, l = matching distance - 3; otherwise l is literal.
// Assume "code" field has at least 64 bits available.
void huffman_translate(uint16 l, uint16 d, uint16 *total_length,
                       uint64 *out_buf, uint64 *outl_buf) {
#pragma HLS PIPELINE II=1
uint16 l_bound[2*L_EXTRA] = {
0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96, 112,128,160,192,224,255,
0,1,2,3,4,5,6,7,9,11,13,15,19,23,27,31,39,47,55,63,79,95,111,127,159,191,223,254,255
};
uint16 d_bound[2*D_EXTRA] = {
1,2,3,4,5,7,9, 13,17,25,33,49,65,97, 129,193,257,385,513,769, 1025,1537,2049,3073,4097,6145,8193, 12289,16385,24577,
1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192,12288,16384,24576,32768
};

uint16 l_extra_bits[L_EXTRA] = {
0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0
};

uint16 d_extra_bits[D_EXTRA] = {
0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13
};

// Static huffman tree

uint16 ltree[572] = {
8, 12,
8, 140,
8, 76,
8, 204,
8, 44,
8, 172,
8, 108,
8, 236,
8, 28,
8, 156,
8, 92,
8, 220,
8, 60,
8, 188,
8, 124,
8, 252,
8, 2,
8, 130,
8, 66,
8, 194,
8, 34,
8, 162,
8, 98,
8, 226,
8, 18,
8, 146,
8, 82,
8, 210,
8, 50,
8, 178,
8, 114,
8, 242,
8, 10,
8, 138,
8, 74,
8, 202,
8, 42,
8, 170,
8, 106,
8, 234,
8, 26,
8, 154,
8, 90,
8, 218,
8, 58,
8, 186,
8, 122,
8, 250,
8, 6,
8, 134,
8, 70,
8, 198,
8, 38,
8, 166,
8, 102,
8, 230,
8, 22,
8, 150,
8, 86,
8, 214,
8, 54,
8, 182,
8, 118,
8, 246,
8, 14,
8, 142,
8, 78,
8, 206,
8, 46,
8, 174,
8, 110,
8, 238,
8, 30,
8, 158,
8, 94,
8, 222,
8, 62,
8, 190,
8, 126,
8, 254,
8, 1,
8, 129,
8, 65,
8, 193,
8, 33,
8, 161,
8, 97,
8, 225,
8, 17,
8, 145,
8, 81,
8, 209,
8, 49,
8, 177,
8, 113,
8, 241,
8, 9,
8, 137,
8, 73,
8, 201,
8, 41,
8, 169,
8, 105,
8, 233,
8, 25,
8, 153,
8, 89,
8, 217,
8, 57,
8, 185,
8, 121,
8, 249,
8, 5,
8, 133,
8, 69,
8, 197,
8, 37,
8, 165,
8, 101,
8, 229,
8, 21,
8, 149,
8, 85,
8, 213,
8, 53,
8, 181,
8, 117,
8, 245,
8, 13,
8, 141,
8, 77,
8, 205,
8, 45,
8, 173,
8, 109,
8, 237,
8, 29,
8, 157,
8, 93,
8, 221,
8, 61,
8, 189,
8, 125,
8, 253,
9, 19,
9, 275,
9, 147,
9, 403,
9, 83,
9, 339,
9, 211,
9, 467,
9, 51,
9, 307,
9, 179,
9, 435,
9, 115,
9, 371,
9, 243,
9, 499,
9, 11,
9, 267,
9, 139,
9, 395,
9, 75,
9, 331,
9, 203,
9, 459,
9, 43,
9, 299,
9, 171,
9, 427,
9, 107,
9, 363,
9, 235,
9, 491,
9, 27,
9, 283,
9, 155,
9, 411,
9, 91,
9, 347,
9, 219,
9, 475,
9, 59,
9, 315,
9, 187,
9, 443,
9, 123,
9, 379,
9, 251,
9, 507,
9, 7,
9, 263,
9, 135,
9, 391,
9, 71,
9, 327,
9, 199,
9, 455,
9, 39,
9, 295,
9, 167,
9, 423,
9, 103,
9, 359,
9, 231,
9, 487,
9, 23,
9, 279,
9, 151,
9, 407,
9, 87,
9, 343,
9, 215,
9, 471,
9, 55,
9, 311,
9, 183,
9, 439,
9, 119,
9, 375,
9, 247,
9, 503,
9, 15,
9, 271,
9, 143,
9, 399,
9, 79,
9, 335,
9, 207,
9, 463,
9, 47,
9, 303,
9, 175,
9, 431,
9, 111,
9, 367,
9, 239,
9, 495,
9, 31,
9, 287,
9, 159,
9, 415,
9, 95,
9, 351,
9, 223,
9, 479,
9, 63,
9, 319,
9, 191,
9, 447,
9, 127,
9, 383,
9, 255,
9, 511,
7, 0,
7, 64,
7, 32,
7, 96,
7, 16,
7, 80,
7, 48,
7, 112,
7, 8,
7, 72,
7, 40,
7, 104,
7, 24,
7, 88,
7, 56,
7, 120,
7, 4,
7, 68,
7, 36,
7, 100,
7, 20,
7, 84,
7, 52,
7, 116,
8, 3,
8, 131,
8, 67,
8, 195,
8, 35,
8, 163
};

uint16 dtree[60] = {
5, 0,
5, 16,
5, 8,
5, 24,
5, 4,
5, 20,
5, 12,
5, 28,
5, 2,
5, 18,
5, 10,
5, 26,
5, 6,
5, 22,
5, 14,
5, 30,
5, 1,
5, 17,
5, 9,
5, 25,
5, 5,
5, 21,
5, 13,
5, 29,
5, 3,
5, 19,
5, 11,
5, 27,
5, 7,
5, 23
};

  // Index into l, d bounds
  uint16 il, id;
  uint16 out_buf1[4];
  uint16 outl_buf1[4];

  uint16 lindex=0, dindex=0;

  // We first translate the (L,D) pair into the deflate code values and extra bits
  uint16 l_code, d_code;
  uint16 l_extra, d_extra, l_extra_len, d_extra_len;

  // Output buffer
  uint16 lcode_outb = 0, dcode_outb = 0, llen_outb = 0, dlen_outb = 0;

  // Compute offset in l_extra regardless, although matching distance may be 0
  for (il = 0; il < L_EXTRA; il++) {
    uint16 ll, lh;
    ll = l_bound[il];
    lh = l_bound[il+L_EXTRA];
    if ((ll <= l) && (l <= lh)) {
      lindex = il;
    }
  }

  // Compute offset in d_extra regardless
  for (id = 0; id < D_EXTRA; id++) {
    uint16 dl, dh;
    dl = d_bound[id];
    dh = d_bound[id+D_EXTRA];
    if ((dl <= d) && (d <= dh)) {
      dindex = id;  // Caller must guarantee d <= 32768
    }
  }

  if (d == 0) {
    l_code = l; l_extra = 0; l_extra_len = 0;
    d_code = 0; d_extra = 0; d_extra_len = 0;
  } else {
    l_code = lindex + 257;
    l_extra = l - l_bound[lindex];
    l_extra_len = l_extra_bits[lindex];

//    cout << "** lcode: " << l_code << " **";

    d_code = dindex;
    d_extra = d - d_bound[dindex];
    d_extra_len = d_extra_bits[dindex];
  }

  // Now we just need to lookup the huffman tree for our code and output them in 
  // the following order: l_code, l_extra, d_code, d_extra
  //
  //    MSB                                                LSB
  //      [d_extra]  |  [d_code]  |  [l_extra]  |  [l_code]
  //
  // Extra bit fields are in big endian order.

  // lcode
  llen_outb = ltree[l_code * 2];
  lcode_outb = ltree[l_code * 2 + 1];

  // dcode: lookup regardless of matching distance
  dlen_outb = dtree[d_code * 2];
  dcode_outb = dtree[d_code * 2 + 1];

  // If it's a literal, then set dcode length to 0
  if (d == 0) {
    dlen_outb = 0;
    d_extra_len = 0;
  }

  // Clear code if code is of length 0
  if (llen_outb == 0) lcode_outb = 0;
  if (dlen_outb == 0) dcode_outb = 0;
  if (l_extra_len == 0) l_extra = 0;
  if (d_extra_len == 0) d_extra = 0;
/*
  // huffmand translate output dump
  fprintf(stderr, "Input (L,D) = (%d(%02x), %d)\n", l, l, d);
  fprintf(stderr, "Tuple (Lcode, Llen, Lextra, Lelen, Dcode, Dlen, Dextra, Delen)"
      " = (%x,%d,%x,%d,%x,%d,%x,%d)\n", lcode_outb, llen_outb, l_extra, 
      l_extra_len, dcode_outb, dlen_outb, d_extra, d_extra_len);
*/
  out_buf1[0] = lcode_outb;
  out_buf1[1] = l_extra;
  out_buf1[2] = dcode_outb;
  out_buf1[3] = d_extra;

  outl_buf1[0] = llen_outb;
  outl_buf1[1] = l_extra_len;
  outl_buf1[2] = dlen_outb;
  outl_buf1[3] = d_extra_len;

  *total_length = outl_buf1[0] + outl_buf1[1]
       + outl_buf1[2] + outl_buf1[3];
  *out_buf = uint16_to_uint64(out_buf1);
  *outl_buf = uint16_to_uint64(outl_buf1);
/*
  cout << "L: " << l << " D: " << d << ";  "
       << outl_buf1[0] << " "
       << outl_buf1[1] << " "
       << outl_buf1[2] << " "
       << outl_buf1[3] << " "
       << endl;
*/
}

