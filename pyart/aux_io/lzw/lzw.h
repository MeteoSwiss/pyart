/*
 * This file is derived from pymetranet
 * https://github.com/eldesradar/pymetranet
 *
 * Copyright (c) 2026, Eldes Radar
 * Licensed under the BSD 3-Clause License.
 *
 * See the LICENSE file in the root of this project for the full license text.
 */

/*lzw.h
*/

int Compress(unsigned char *inputaddr, int insize, unsigned char *outputaddr, int outsize);
int Expand(unsigned char *inputaddr, int insize, unsigned char *outputaddr, int outsize);
